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
import requests  # 新增requests用于API调用

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

# 语音合成函数
def text_to_speech(text):
    from ConfigManager import ConfigManager
    """将文本转换为语音并返回音频文件路径"""    
    conf = ConfigManager()
    api_key = conf.get_text_model_config().get("api_key", "")
    if not api_key or not text:
        logger.warning("API密钥或文本为空，无法生成语音")
        return None
    
    try:
        # 清理文本中的特殊标记
        clean_text = re.sub(r'\([^)]*\)', '', text)
        
        # 准备请求硅基流动API
        payload = {
            "model": "FunAudioLLM/CosyVoice2-0.5B",
            "input": clean_text,
            "voice": "FunAudioLLM/CosyVoice2-0.5B:diana"
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 请求语音合成
        response = requests.post(
            "https://api.siliconflow.cn/v1/audio/speech", 
            json=payload, 
            headers=headers,
            timeout=60  # 增加超时时间
        )
        response.raise_for_status()
        
        # 创建临时音频文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(response.content)
            audio_file_path = f.name
            logger.info(f"语音合成成功，保存到: {audio_file_path}")
        
        return audio_file_path
    except Exception as e:
        logger.error(f"语音合成失败: {str(e)}")
        return None

# 创建 Gradio 界面
with gr.Blocks(title="Python语言与人工智能应用大作业-严小希的工作间") as demo:
    # 状态变量
    username = gr.State("cxxdgc")
    current_page = gr.State(0)
    chat_history = gr.State([])
    phone_clues_state = gr.State([]) # 当前所有的电话号码线索
    plate_clues_state = gr.State([]) # 当前所有的车牌号线索
    current_audio_file = gr.State(None)  # 当前语音文件路径

    # 自动摘抄的状态变量
    input_text = gr.State("")
    excerpts_state = gr.State([])
    
    # 整个应用布局 - 使用行布局并指定比例
    with gr.Row():
        # 左侧导航栏 - 设置为1/7宽度
        with gr.Column(scale=1, min_width=100):
            with gr.Column(elem_classes="panel", variant="panel"):
                gr.Markdown("### 导航菜单")
                home_btn = gr.Button("首页", size="sm", variant="primary")
                feature1_btn = gr.Button("自动批注", size="sm",visible=False)
                feature2_btn = gr.Button("自动摘抄", size="sm")
                geolocate_btn = gr.Button("街景定位", size="sm")
                yan_btn = gr.Button("关于严小希", size="sm")
                settings_btn = gr.Button("设置", size="sm",visible=False)
                
                gr.Markdown("---")
                gr.Markdown(f"**用户**: cxxdgc")
                logout_btn = gr.Button("退出登录", size="sm", variant="stop")
        
        # 右侧内容区 - 设置为6/7宽度
        with gr.Column(scale=6):
            # 页面容器
            with gr.Column(visible=True, elem_classes="panel") as home_container:
                # 聊天界面
                gr.Markdown("## 你好呀~我是严小希")
                gr.Markdown("与严小希交流...")
                
                chatbot = gr.Chatbot(height=500, label="严小希对话")
                msg = gr.Textbox(label="请输入消息", placeholder="输入您的对话后按Enter发送...")
                
                # 新增语音输出功能
                with gr.Row():
                    with gr.Column(scale=4):
                        with gr.Row():
                            clear_btn = gr.Button("清空对话")
                            file_upload = gr.UploadButton("📁 上传文件", file_types=["text", ".json", ".pdf", ".docx"])
                        with gr.Row():
                            # 语音输出按钮
                            voice_output_btn = gr.Button("🔊 语音输出", size="sm")
                    with gr.Column(scale=1):
                        # 语音输出状态指示器
                        voice_status = gr.Textbox(label="语音状态", interactive=False, placeholder="准备中...", visible=False)
                # 语音播放器
                audio_player = gr.Audio(label="严小希语音", type="filepath", visible=False)
            
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

            with gr.Column(visible=False, elem_classes="panel") as geolocate_container:
                gr.Markdown("## 🌍 街景地理定位器")
                gr.Markdown("上传街景图片，AI将识别地理位置特征")
                
                with gr.Row():
                    with gr.Column():
                        # 上传组件
                        image_input = gr.Image(label="上传街景图片", type="filepath")
                        geolocate_button = gr.Button("开始定位")
                        
                        # 设置折叠面板
                        with gr.Accordion("高级设置", open=False, visible= False):
                            openai_key_input = gr.Textbox(
                                label="OpenAI API Key",
                                type="password",
                                placeholder="输入你的OpenAI API Key（可选）",
                                info="如果使用硅基流动API，可以留空"
                            )
                            openai_model = gr.Dropdown(
                                label="选择模型",
                                choices=["gpt-4-vision-preview", "gpt-4-turbo"],
                                value="gpt-4-vision-preview"
                            )
                            map_provider = gr.Dropdown(
                                label="地图提供商",
                                choices=["百度地图", "谷歌地图", "OpenStreetMap"],
                                value="百度地图"
                            )
                        
                        # 新增的线索展示板块
                        with gr.Column(elem_classes="panel"):
                            gr.Markdown("### 🔎 可能的线索")
                            
                            # 电话号码线索
                            with gr.Accordion("电话号码线索", open=False):
                                with gr.Row():
                                    phone_clues = gr.Dropdown(
                                        choices=[], 
                                        label="图中出现的电话号",
                                        interactive=True,
                                        multiselect=True
                                    )
                                    # 添加手动输入区域
                                    manual_phone_input = gr.Textbox(
                                        label="手动输入电话号",
                                        placeholder="输入自定义电话号..."
                                    )
                                
                                with gr.Row():
                                    phone_location = gr.Textbox(
                                        label="电话号码归属地",
                                        lines=1, 
                                        interactive=False, 
                                        placeholder="点击号码查看归属地信息..."
                                    )
                                    # 添加手动添加按钮
                                    # add_phone_btn = gr.Button("添加号码", size="sm")
                                with gr.Row():
                                    phone_lookup_btn = gr.Button("查找归属地", size="sm")
                                    clear_phone_btn = gr.Button("清空号码", size="sm", variant="stop")
                            
                            # 车牌号线索
                            with gr.Accordion("车牌号线索", open=False):
                                with gr.Row():
                                    plate_clues = gr.Dropdown(
                                        choices=[], 
                                        label="图中出现的车牌号",
                                        interactive=True,
                                        multiselect=True,
                                    )
                                    # 手动输入区域
                                    manual_plate_input = gr.Textbox(
                                        label="手动输入车牌号",
                                        placeholder="输入自定义车牌号..."
                                    )
                                
                                with gr.Row():
                                    plate_location = gr.Textbox(
                                        label="车牌号归属地",
                                        lines=1, 
                                        interactive=False, 
                                        placeholder="点击车牌号查看归属地信息..."
                                    )
                                    # 添加手动添加按钮
                                    # add_plate_btn = gr.Button("添加车牌", size="sm")
                                with gr.Row():
                                    plate_lookup_btn = gr.Button("查找归属地", size="sm")
                                    clear_plate_btn = gr.Button("清空车牌", size="sm", variant="stop")
                            
                            # 其他线索
                            with gr.Accordion("其他线索", open=False):
                                other_clues = gr.Textbox(
                                    label="其他有用线索",
                                    lines=3, 
                                    interactive=False, 
                                    placeholder="其他可能有助于定位的信息..."
                                )
                    
                    with gr.Column():
                        # 结果显示区域
                        map_html = gr.HTML(label="地图预览", value="<div style='text-align:center;color:#888;'>地图将在此显示</div>")
                        text_output = gr.Textbox(label="位置信息", lines=3, interactive=False)
                        
                        # 加载状态
                        loading_overlay_geo = gr.HTML(visible=False, value="""
                            <div class="loading-overlay" style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: rgba(255, 255, 255, 0.8); z-index: 999; display: flex; align-items: center; justify-content: center; pointer-events: none;">
                                <div class="loader" style="border: 4px solid #f3f3f3; border-top: 4px solid #8B0012; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite;"></div>
                            </div>
                            <style>
                            @keyframes spin {
                                0% { transform: rotate(0deg); }
                                100% { transform: rotate(360deg); }
                            }
                            </style>
                        """)
                        
                        # 新增的原因描述板块
                        with gr.Column(elem_classes="panel"):
                            gr.Markdown("### 🔍 原因描述")
                            reasoning_output = gr.Textbox(
                                label="AI分析原因", 
                                lines=5, 
                                interactive=False, 
                                placeholder="AI将在这里解释为什么做出这个定位决策..."
                            )

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
                        gr.Markdown("[获取硅基流动api↗](https://cloud.siliconflow.cn/me/account/ak)")
                save_btn = gr.Button("保存设置")
                status = gr.Markdown("")
    
    # 页面切换函数
    def show_page(page_idx):
        return [gr.update(visible=page_idx==i) for i in range(6)]
    
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

    # 语音输出按钮的点击事件
    voice_output_btn.click(
        # 首先显示加载状态
        lambda: [gr.update(visible=True), gr.update(value="正在生成语音...")],
        outputs=[audio_player, voice_status]
    ).then(
        # 获取最近一条回复并生成语音
        lambda chat_history: [
            text_to_speech(
                chat_history[-1][1] if chat_history else "你好呀，我是严小希~"),
            gr.update(visible=True),
            gr.update(value="语音已生成，点击播放按钮收听")
        ],
        inputs=[chat_history],  # 需要添加settings_state变量
        outputs=[audio_player, audio_player, voice_status]
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

    
    # 地理位置定位相关函数
    import grolocate
    geolocate_button.click(
        grolocate.process_geolocation,
        inputs=[image_input, openai_key_input, openai_model, map_provider],
        outputs=[loading_overlay_geo, map_html, text_output, reasoning_output, phone_clues, plate_clues, other_clues]
    )

    
    # 添加线索按钮事件
    # add_phone_btn.click(
    #     grolocate.add_custom_clue,
    #     inputs=[manual_phone_input, phone_clues_state],
    #     outputs=[phone_clues_state, phone_clues]
    # ).then(
    #     lambda: "",  # 清空输入框
    #     outputs=[manual_phone_input]
    # )

    # add_plate_btn.click(
    #     grolocate.add_custom_clue,
    #     inputs=[manual_plate_input, plate_clues_state],
    #     outputs=[plate_clues_state, plate_clues]
    # ).then(
    #     lambda: "",  # 清空输入框
    #     outputs=[manual_plate_input]
    # )

    # 清空线索按钮事件
    clear_phone_btn.click(
        grolocate.clear_clues,
        outputs=[phone_clues_state, phone_clues]
    )

    clear_plate_btn.click(
        grolocate.clear_clues,
        outputs=[plate_clues_state, plate_clues]
    )
    
    # 电话号码线索点击事件
    phone_lookup_btn.click(
        lambda manual_phone, phone_options: 
            grolocate.lookup_phone_location(manual_phone.strip()) if manual_phone.strip() 
            else grolocate.lookup_phone_location(phone_options[0].split(" ")[0]) if phone_options and "未识别" not in phone_options[0] 
            else "请输入或选择有效的电话号码",
        inputs=[manual_phone_input, phone_clues],
        outputs=phone_location
    )
    
    # 车牌号线索点击事件
    plate_lookup_btn.click(
        lambda manual_plate, plate_options: 
            grolocate.lookup_plate_location(manual_plate.strip()) if manual_plate.strip() 
            else grolocate.lookup_plate_location(plate_options[0].split(" ")[0]) if plate_options and "未识别" not in plate_options[0] 
            else "请输入或选择有效的车牌号",
        inputs=[manual_plate_input, plate_clues],
        outputs=plate_location
    )
    
    # 导航按钮事件
    home_btn.click(lambda: [0, *show_page(0)], outputs=[current_page, home_container, feature1_container, feature2_container, yan_container, settings_container, geolocate_container])
    feature1_btn.click(lambda: [1, *show_page(1)], outputs=[current_page, home_container, feature1_container, feature2_container, yan_container, settings_container, geolocate_container])
    feature2_btn.click(lambda: [2, *show_page(2)], outputs=[current_page, home_container, feature1_container, feature2_container, yan_container, settings_container, geolocate_container])
    yan_btn.click(lambda: [3, *show_page(3)], outputs=[current_page, home_container, feature1_container, feature2_container, yan_container, settings_container, geolocate_container])
    settings_btn.click(lambda: [4, *show_page(4)], outputs=[current_page, home_container, feature1_container, feature2_container, yan_container, settings_container, geolocate_container])
    geolocate_btn.click(lambda: [5, *show_page(5)], outputs=[current_page, home_container, feature1_container, feature2_container, yan_container, settings_container, geolocate_container])
    
    # 登出事件
    def logout():
        raise gr.Error("当前为测试版demo，用户系统暂未开放！")   
    logout_btn.click(logout,outputs=None)

    
    # 初始加载显示首页
    demo.load(lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)],
              outputs=[home_container, feature1_container, feature2_container, yan_container, settings_container, geolocate_container])

# 启动应用
if __name__ == "__main__":
    demo.launch()