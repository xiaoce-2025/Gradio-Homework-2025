# -*- coding: utf-8 -*-

import random
import gradio as gr
import logging
from datetime import datetime
import tempfile
import os
import cv2
import re
import json
import html
from openai import OpenAI
from paddleocr import PaddleOCR

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s]%(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)



# 关于严小希的HTML内容
from YanxxPage import Yanxx_Page
def yan_page_html():
    return Yanxx_Page()

# 创建 Gradio 界面
with gr.Blocks(title="严小希", css=".panel {border-radius: 10px; padding: 15px;}") as demo:
    # 状态变量
    username = gr.State("cxxdgc")
    current_page = gr.State(0)
    chat_history = gr.State([])

    # 自动摘抄的状态变量
    input_text = gr.State("")
    excerpts_state = gr.State([])
    
    # 整个应用布局 - 使用行布局并指定比例
    with gr.Row():
        # 左侧导航栏 - 设置为1/5宽度
        with gr.Column(scale=1, min_width=200):
            with gr.Column(elem_classes="panel", variant="panel"):
                gr.Markdown("### 导航菜单")
                home_btn = gr.Button("首页", size="sm", variant="primary")
                feature1_btn = gr.Button("自动批注", size="sm")
                feature2_btn = gr.Button("自动摘抄", size="sm")
                yan_btn = gr.Button("关于严小希", size="sm")
                settings_btn = gr.Button("设置", size="sm")
                
                gr.Markdown("---")
                gr.Markdown(f"**用户**: cxxdgc")
                logout_btn = gr.Button("退出登录", size="sm", variant="stop")
        
        # 右侧内容区 - 设置为4/5宽度
        with gr.Column(scale=4):
            # 页面容器
            with gr.Column(visible=True, elem_classes="panel") as home_container:
                # 聊天界面
                gr.Markdown("## 你好呀~我是严小希")
                gr.Markdown("与严小希交流...")
                
                chatbot = gr.Chatbot(height=500, label="严小希对话")
                msg = gr.Textbox(label="请输入消息", placeholder="输入您的对话后按Enter发送...")
                with gr.Row():
                    clear_btn = gr.Button("清空对话")
                    file_upload = gr.UploadButton("📁 上传文件", file_types=["text", ".json", ".pdf", ".docx"])
                file_output = gr.Markdown()
            
            # 自动批注页面
            with gr.Column(visible=False, elem_classes="panel") as feature1_container:
                gr.Markdown("## 自动批注功能")
                gr.Markdown("由于网络安全问题，该功能正在上线校验中，暂时不可使用~")
                
                with gr.Row():
                    text_input = gr.Textbox(label="输入文本", lines=10, placeholder="在此输入要批注的文本...")
                    file_input = gr.File(label="或上传文档", file_types=[".txt", ".pdf", ".docx"])
                
                generate_btn = gr.Button("开始批注")
                output_area = gr.Textbox(label="批注结果", interactive=False, lines=15)
            
            # 自动摘抄页面
            with gr.Column(visible=False, elem_classes="panel") as feature2_container:
                gr.Markdown("## 自动化摘抄")
                
                # 加载遮罩层
                loading_overlay = gr.HTML(visible=False, value="""
                    <div class="loading-overlay" style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: rgba(255, 255, 255, 0.8); z-index: 999; display: flex; align-items: center; justify-content: center;">
                        <div class="loader" style="border: 4px solid #f3f3f3; border-top: 4px solid #8B0012; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite;"></div>
                    </div>
                    <style>
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                    </style>
                """)
                
                with gr.Column():
                    text_input_f2 = gr.Textbox(label="输入文本", lines=10, placeholder="粘贴或输入您的摘抄内容...")
                    
                    # 使用可见的文件上传组件
                    picture_upload = gr.File(label="图片OCR", file_types=["image"], visible=True)
                    
                    with gr.Row():
                        # OCR按钮现在只是隐藏文件上传组件的占位符
                        # 实际文件上传由用户直接点击文件上传组件触发
                        ocr_btn = gr.Button("📷 选择图片", visible=False)
                        extract_btn = gr.Button("🚀 摘抄，启动！")
                        export_btn = gr.Button("📤 导出摘抄内容")
                        clear_btn_f2 = gr.Button("🗑 清空内容")
                    
                    # 结果导出
                    download_component = gr.File(visible=False, label="导出结果")  # 新增下载组件

                    # 摘抄结果展示
                    excerpts_display = gr.HTML(value="<div class='excerpts-list'></div>")

            
            # 关于严小希页面
            with gr.Column(visible=False, elem_classes="panel") as yan_container:
                gr.HTML(yan_page_html())
            
            # 设置页面
            with gr.Column(visible=False, elem_classes="panel") as settings_container:
                gr.Markdown("## 系统设置")
                gr.Markdown("由于网络安全问题，该正在上线校验中，暂时不可使用~")
                
                with gr.Row():
                    with gr.Column(min_width=300):
                        gr.Markdown("### 界面设置")
                        dark_mode = gr.Checkbox(label="深色模式")
                        notification = gr.Checkbox(label="启用通知")
                    
                    with gr.Column(min_width=300):
                        gr.Markdown("### AI设置")
                        ai_temperature = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.7, label="创造力")
                        ai_max_tokens = gr.Slider(minimum=100, maximum=2000, step=100, value=1024, label="最大生成长度")
                        ai_key = gr.Textbox(label="硅基流动api",placeholder="请输入硅基流动api",lines=1)
                
                save_btn = gr.Button("保存设置")
                status = gr.Markdown("")
    
    # 页面切换函数
    def show_page(page_idx):
        return [gr.update(visible=page_idx==i) for i in range(5)]
    
    # 聊天页面的事件
    from YanxxDialog import Yanxx_respond

    msg.submit(
        Yanxx_respond,
        inputs=[msg, chat_history],  # 输入：消息内容，聊天历史
        outputs=[msg, chatbot, chat_history]  # 输出：清空输入框，更新聊天框，更新历史状态
    )
    
    clear_btn.click(
        lambda: [[], []],
        outputs=[chatbot, chat_history]
    )
    
    file_upload.upload(
        lambda file: f"已收到文件: {file.name}",
        inputs=[file_upload],
        outputs=file_output
    )
    
    # 批注页面的事件
    generate_btn.click(
        lambda text, file: "这是生成的批注示例：\n\n- 第一点批注\n- 第二点批注\n- 第三点批注",
        inputs=[text_input, file_input],
        outputs=output_area
    )
    
    
    # 设置页面的事件
    save_btn.click(
        lambda dark, notify, temp, tokens: "设置已保存成功！",
        inputs=[dark_mode, notification, ai_temperature, ai_max_tokens],
        outputs=status
    )

    # 自动摘抄功能函数
    from TextActracts import process_ocr,export_excerpts,extract_excerpts
    
    # 自动摘抄页面事件
    picture_upload.upload(
        lambda: gr.update(visible=True),  # 显示加载动画
        outputs=[loading_overlay]
    ).then(
        process_ocr,  # 进行OCR处理
        inputs=[picture_upload],  # 从文件上传组件获取文件
        outputs=[text_input_f2]
    ).then(
        lambda: gr.update(visible=False),  # 隐藏加载动画
        outputs=[loading_overlay]
    )
        
    from ConfigManager import ConfigManager
    conf = ConfigManager()
    SILICONFLOW_API_KEY = conf.get_text_model_config()["api_key"]
    SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/"
    MODEL_NAME = conf.get_text_model_config()["name"]

    extract_btn.click(
        lambda: gr.update(visible=True),
        outputs=[loading_overlay]
    ).then(
        extract_excerpts,
        inputs=[text_input_f2, gr.State({
            "ModelName": MODEL_NAME,
            "apiKey": SILICONFLOW_API_KEY,
            "apiUrl": SILICONFLOW_API_URL,
            "maxToken": 4000,
            "ModelTemperature": 0.7
        })],
        outputs=[excerpts_display, excerpts_state]
    ).then(
        lambda: gr.update(visible=False),
        outputs=[loading_overlay]
    )
    
    export_btn.click(
        export_excerpts,
        inputs=[excerpts_state],
        outputs=[status, download_component]  # 改为使用下载组件
    ).then(
        lambda: gr.update(visible=True),  # 显示下载组件
        outputs=[download_component]
    )
    
    clear_btn_f2.click(
        lambda: ["", "", []],
        outputs=[text_input_f2, excerpts_display, excerpts_state]
    )
    
    # 导航按钮事件
    home_btn.click(lambda: [0, *show_page(0)], outputs=[current_page, home_container, feature1_container, feature2_container, yan_container, settings_container])
    feature1_btn.click(lambda: [1, *show_page(1)], outputs=[current_page, home_container, feature1_container, feature2_container, yan_container, settings_container])
    feature2_btn.click(lambda: [2, *show_page(2)], outputs=[current_page, home_container, feature1_container, feature2_container, yan_container, settings_container])
    yan_btn.click(lambda: [3, *show_page(3)], outputs=[current_page, home_container, feature1_container, feature2_container, yan_container, settings_container])
    settings_btn.click(lambda: [4, *show_page(4)], outputs=[current_page, home_container, feature1_container, feature2_container, yan_container, settings_container])
    
    # 登出事件
    
    # 初始加载显示首页
    demo.load(lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)],
              outputs=[home_container, feature1_container, feature2_container, yan_container, settings_container])

# 启动应用
if __name__ == "__main__":
    demo.launch()