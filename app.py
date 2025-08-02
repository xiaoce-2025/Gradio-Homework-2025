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
with gr.Blocks(
    title="严小希", 
    css="""
        /* 全局样式覆盖 */
        body {
            margin: 0;
            padding: 0;
            overflow: hidden;
        }
        #component-0, .app {
            max-width: none !important;
            min-height: 100vh;
            margin: 0;
            padding: 0;
        }
        .contain { /* Gradio 的主要容器 */
            max-width: none !important;
            min-width: 100% !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        .panel {
            border-radius: 10px; 
            padding: 15px;
        }
        #content-box { /* 右侧内容区 */
            padding: 15px 20px;
        }
        .gradio-container {
            min-height: 100vh;
        }
        .h-full {
            height: 100%;
        }
    """) as demo:
    # 状态变量
    username = gr.State("cxxdgc")
    current_page = gr.State(0)
    chat_history = gr.State([])

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
                feature1_btn = gr.Button("自动批注", size="sm")
                feature2_btn = gr.Button("自动摘抄", size="sm")
                geolocate_btn = gr.Button("街景定位", size="sm")
                yan_btn = gr.Button("关于严小希", size="sm")
                settings_btn = gr.Button("设置", size="sm")
                
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

            with gr.Column(visible=False, elem_classes="panel") as geolocate_container:
                gr.Markdown("## 🌍 街景地理定位器")
                gr.Markdown("上传街景图片，AI将识别地理位置特征")
                
                with gr.Row():
                    with gr.Column():
                        # 上传组件
                        image_input = gr.Image(label="上传街景图片", type="filepath")
                        geolocate_button = gr.Button("开始定位")
                        
                        # 设置折叠面板
                        with gr.Accordion("高级设置", open=False):
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
                                phone_clues = gr.Dropdown(
                                    choices=[], 
                                    label="图中出现的手机号",
                                    interactive=True,
                                    multiselect=True,
                                    info="点击电话号码可以查看归属地信息"
                                )
                                phone_location = gr.Textbox(
                                    label="电话号码归属地",
                                    lines=1, 
                                    interactive=False, 
                                    placeholder="点击号码查看归属地信息..."
                                )
                                phone_lookup_btn = gr.Button("查找归属地", size="sm")
                            
                            # 车牌号线索
                            with gr.Accordion("车牌号线索", open=False):
                                plate_clues = gr.Dropdown(
                                    choices=[], 
                                    label="图中出现的车牌号",
                                    interactive=True,
                                    multiselect=True,
                                )
                                plate_location = gr.Textbox(
                                    label="车牌号归属地",
                                    lines=1, 
                                    interactive=False, 
                                    placeholder="点击车牌号查看归属地信息..."
                                )
                                plate_lookup_btn = gr.Button("查找归属地", size="sm")
                            
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
                        
                        
                        
                        # 示例图片
                        gr.Examples(
                            examples=[
                                ["demo_street1.jpg"],
                                ["demo_street2.jpg"]
                            ],
                            inputs=image_input,
                            outputs=[map_html, text_output],
                            fn=None,  # 将使用单独的处理函数
                            cache_examples=False
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

    # 新增街景定位功能函数
    def geolocate_image(image_path, api_key, model, map_provider):
        """使用大模型分析图像并获取地理位置"""
        try:
            # 如果没有提供OpenAI API Key，使用硅基流动API
            if not api_key:
                from ConfigManager import ConfigManager
                conf = ConfigManager()
                SILICONFLOW_API_KEY = conf.get_text_model_config().get("api_key", "")
                SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/"
                model_name = conf.get_text_model_config().get("name", "")
                
                if not SILICONFLOW_API_KEY:
                    return None, "⚠️ 错误：未提供API密钥，请设置硅基流动API或提供OpenAI API Key"
                
                # 使用硅基流动API
                headers = {
                "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
                "Content-Type": "application/json"
                }
                    
                
                import base64
                # 读取并编码图像
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                
                # 构建请求体
                payload = {
                    "model": "THUDM/GLM-4.1V-9B-Thinking",  # 指定视觉模型
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "详细分析这张街景图像中的所有文字、路标、建筑特征和任何地理标识信息。"
                                            "并根据这些信息推断可能的地理位置，精确到城市。"
                                            "同时请识别图像中出现的手机号和车牌号（如果有的话）。"
                                            "输出格式为JSON: {'location': '国家 城市 具体位置', 'reasoning': '详细的原因解释', 'phone_numbers': ['11111111111', '22222222222'], 'license_plates': ['沪A12345', '京B67890']}"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 800
                }
                import requests
                # 发送请求
                response = requests.post(
                    f"{SILICONFLOW_API_URL}chat/completions",
                    headers=headers,
                    json=payload
                )
                
                # 检查响应状态
                if response.status_code != 200:
                    error_msg = f"API请求失败: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    return None, f"⚠️ API错误: {error_msg}"
                
                # 解析响应
                result = response.json()
                if "choices" not in result or not result["choices"]:
                    return None, "⚠️ API未返回有效结果"
                
                # 提取模型输出
                content = result["choices"][0]["message"]["content"]
                print(content)
                # 尝试解析为JSON
                try:
                    # 提取JSON部分（去除多余文本）
                    start_idx = content.find('{')
                    end_idx = content.rfind('}') + 1
                    json_content = content[start_idx:end_idx]
                    
                    data = json.loads(json_content)
                    
                    location = data.get('location', '未知位置')
                    reasoning = data.get('reasoning', '没有提供分析原因')
                    phone_numbers = data.get('phone_numbers', [])
                    license_plates = data.get('license_plates', [])
                except Exception as e:
                    logger.error(f"JSON解析错误: {str(e)}")
                    logger.info(f"原始响应内容: {content}")
                    
                    # 尝试直接提取位置
                    if "," in content:
                        country, city, *location_parts = content.split(",", 2)
                        location = f"{country.strip()}{city.strip()}"
                    else:
                        location = content
                    
                    reasoning = "无法解析AI的分析结果"
                    phone_numbers = []
                    license_plates = []

            else:
                location = "暂不支持openai大模型"
                reasoning = ""
                phone_numbers = []
                license_plates = []
            
            # 根据地图提供商生成地图预览
            map_html = f"""
            <iframe src="https://map.qq.com/?type=poi&what={location}"
                    width="800" height="400" frameborder="0" style="border:0;"></iframe>
            <p style="text-align:center;margin-top:10px;color:#666;">{location}</p>
            """
            
            return map_html, location, reasoning, phone_numbers, license_plates
        
        except Exception as e:
            logger.error(f"地理定位失败: {str(e)}")
            error_message = f"⚠️ 处理失败: {str(e)}"
            if "Unauthorized" in str(e):
                error_message = "⚠️ API密钥无效，请检查API Key"
            elif "rate limit" in str(e).lower():
                error_message = "⚠️ 超出API调用限制，请稍后再试"
            return None, error_message, "", [], []

    # 查找电话号码归属地的函数
    def lookup_phone_location(phone_number):
        """查找电话号码归属地（模拟功能）"""
        # 在真实应用中，这里应调用归属地查询API
        prefix = phone_number[:3]
        if prefix == '130':
            return f"{phone_number} - 中国联通 - 北京"
        elif prefix == '138':
            return f"{phone_number} - 中国移动 - 上海"
        elif prefix == '189':
            return f"{phone_number} - 中国电信 - 广州"
        else:
            return f"{phone_number} - 归属地未知"

    # 查找车牌号归属地的函数
    def lookup_plate_location(plate_number):
        """查找车牌号归属地（模拟功能）"""
        # 在真实应用中，这里应调用车牌归属查询API
        if plate_number.startswith('京'):
            return f"{plate_number} - 北京市"
        elif plate_number.startswith('沪'):
            return f"{plate_number} - 上海市"
        elif plate_number.startswith('粤'):
            return f"{plate_number} - 广东省"
        else:
            return f"{plate_number} - 归属地未知"

    # 街景定位按钮的事件
    def process_geolocation(image_path, openai_key, model, map_provider):
        # 显示加载状态
        yield gr.update(visible=True), None, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        
        # 处理图像
        map_html, location, reasoning, phone_numbers, license_plates = geolocate_image(image_path, openai_key, model, map_provider)
        
        # 格式化线索展示
        phone_options = [f"{phone} (点击查看)" for phone in phone_numbers] if phone_numbers else ["未识别到电话号码"]
        plate_options = [f"{plate} (点击查看)" for plate in license_plates] if license_plates else ["未识别到车牌号"]
        
        # 隐藏加载状态
        yield gr.update(visible=False), map_html, gr.update(value=location), gr.update(value=reasoning), \
            gr.update(choices=phone_options, value=[]), \
            gr.update(choices=plate_options, value=[]), \
            gr.update(interactive=True)

    geolocate_button.click(
        process_geolocation,
        inputs=[image_input, openai_key_input, openai_model, map_provider],
        outputs=[loading_overlay_geo, map_html, text_output, reasoning_output, phone_clues, plate_clues, other_clues]
    )
    
    # 电话号码线索点击事件
    phone_lookup_btn.click(
        lambda phone_options: lookup_phone_location(phone_options[0].split(" ")[0]) if phone_options and "未识别" not in phone_options[0] else "请选择有效的电话号码",
        inputs=[phone_clues],
        outputs=phone_location
    )
    
    # 车牌号线索点击事件
    plate_lookup_btn.click(
        lambda plate_options: lookup_plate_location(plate_options[0].split(" ")[0]) if plate_options and "未识别" not in plate_options[0] else "请选择有效的车牌号",
        inputs=[plate_clues],
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
    
    # 初始加载显示首页
    demo.load(lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)],
              outputs=[home_container, feature1_container, feature2_container, yan_container, settings_container, geolocate_container])

# 启动应用
if __name__ == "__main__":
    demo.launch()