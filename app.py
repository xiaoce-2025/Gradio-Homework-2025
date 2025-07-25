import random
import gradio as gr
import json
import requests
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 聊天功能配置
from ConfigManager import ConfigManager
conf = ConfigManager()
SILICONFLOW_API_KEY = conf.get_text_model_config()["api_key"]
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL_NAME = conf.get_text_model_config()["name"]

def LLM_response(message, history):
    """流式生成响应的生成器函数"""
    # 构建消息历史
    messages = []
    for human, assistant in history:
        if (human):
            messages.append({"role": "user", "content": human})
        if (assistant):
            messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": message})

    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1024,
        "stream": True
    }
    logging.info(SILICONFLOW_API_URL)
    logging.info(SILICONFLOW_API_KEY)
    logging.info(MODEL_NAME)
    logging.info("LLM已转发提交")
    logging.info(str(messages))

    try:
        with requests.post(
            SILICONFLOW_API_URL,
            headers=headers,
            json=payload,
            stream=True
        ) as response:
            if response.status_code != 200:
                yield f"API错误: {response.status_code} - {response.text}"
                return
                
            accumulated_text = ""
            for chunk in response.iter_lines():
                if chunk:
                    decoded_chunk = chunk.decode('utf-8')
                    if decoded_chunk.startswith("data:"):
                        json_data = decoded_chunk[5:].strip()
                        if json_data == "[DONE]":
                            return
                        # 解析思考内容
                        try:
                            data = json.loads(json_data)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("reasoning_content", "")
                                if content:
                                    accumulated_text += content
                                    yield accumulated_text
                        except Exception as e:
                            # 解析错误时继续处理下一个chunk
                            pass
                        
                        # 解析回复内容
                        try:
                            data = json.loads(json_data)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    accumulated_text += content
                                    yield accumulated_text
                        except Exception as e:
                            # 解析错误时继续处理下一个chunk
                            continue
    except Exception as e:
        yield f"请求失败: {str(e)}"

# 关于严小希的HTML内容
from YanxxPage import Yanxx_Page
def yan_page_html():
    return Yanxx_Page()

# 创建 Gradio 界面
with gr.Blocks(title="智能助手应用", css=".panel {border-radius: 10px; padding: 15px;}") as demo:
    # 状态变量
    username = gr.State("cxxdgc")
    current_page = gr.State(0)
    chat_history = gr.State([])
    
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
                gr.Markdown("## 智能助手聊天界面")
                gr.Markdown("与AI助手交流，提问问题或进行对话")
                
                chatbot = gr.Chatbot(height=500, label="AI助手")
                msg = gr.Textbox(label="请输入消息", placeholder="输入您的问题后按Enter发送...")
                with gr.Row():
                    clear_btn = gr.Button("清空对话")
                    file_upload = gr.UploadButton("📁 上传文件", file_types=["text", ".json", ".pdf", ".docx"])
                file_output = gr.Markdown()
            
            # 自动批注页面
            with gr.Column(visible=False, elem_classes="panel") as feature1_container:
                gr.Markdown("## 自动批注功能")
                gr.Markdown("上传文本或文档，系统会自动添加批注")
                
                with gr.Row():
                    text_input = gr.Textbox(label="输入文本", lines=10, placeholder="在此输入要批注的文本...")
                    file_input = gr.File(label="或上传文档", file_types=[".txt", ".pdf", ".docx"])
                
                generate_btn = gr.Button("开始批注")
                output_area = gr.Textbox(label="批注结果", interactive=False, lines=15)
            
            # 自动摘抄页面
            with gr.Column(visible=False, elem_classes="panel") as feature2_container:
                gr.Markdown("## 自动摘抄功能")
                gr.Markdown("上传文档，系统会自动提取关键内容和摘要")
                
                file_input_f2 = gr.File(label="上传文档", file_types=[".pdf", ".docx", ".txt"])
                extract_btn = gr.Button("开始提取")
                
                with gr.Row():
                    key_points = gr.Textbox(label="关键要点", interactive=False, lines=10)
                    summary = gr.Textbox(label="文章摘要", interactive=False, lines=10)
            
            # 关于严小希页面
            with gr.Column(visible=False, elem_classes="panel") as yan_container:
                gr.HTML(yan_page_html())
            
            # 设置页面
            with gr.Column(visible=False, elem_classes="panel") as settings_container:
                gr.Markdown("## 系统设置")
                
                with gr.Row():
                    with gr.Column(min_width=300):
                        gr.Markdown("### 界面设置")
                        dark_mode = gr.Checkbox(label="深色模式")
                        notification = gr.Checkbox(label="启用通知")
                    
                    with gr.Column(min_width=300):
                        gr.Markdown("### AI设置")
                        ai_temperature = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.7, label="创造力")
                        ai_max_tokens = gr.Slider(minimum=100, maximum=2000, step=100, value=1024, label="最大生成长度")
                
                save_btn = gr.Button("保存设置")
                status = gr.Markdown("")
    
    # 页面切换函数
    def show_page(page_idx):
        return [gr.update(visible=page_idx==i) for i in range(5)]
    
    # 聊天页面的事件
    # 添加缺失的聊天响应函数
    def respond(message, history):
        """处理用户消息并返回AI的流式响应"""
        # 初始化AI回复为空
        history = history.copy()  # 创建副本避免直接修改原始状态
        history.append((message, ""))  # 添加新消息（AI回复为空）
        
        # 第一次更新：显示用户消息和空白的AI回复
        yield "", history, history
        
        full_response = ""
        # 调用LLM_response生成器获取流式响应
        for token in LLM_response(message, history[:-1]):  # 传入当前消息前的历史
            full_response = token
            # 更新最后一条AI回复内容
            history[-1] = (message, full_response)
            yield "", history, history

    
    msg.submit(
        respond,
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
    
    # 摘抄页面的事件
    extract_btn.click(
        lambda file: ("关键要点：\n1. 要点一\n2. 要点二\n3. 要点三", "文章摘要：\n本文主要讨论了..."),
        inputs=[file_input_f2],
        outputs=[key_points, summary]
    )
    
    # 设置页面的事件
    save_btn.click(
        lambda dark, notify, temp, tokens: "设置已保存成功！",
        inputs=[dark_mode, notification, ai_temperature, ai_max_tokens],
        outputs=status
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