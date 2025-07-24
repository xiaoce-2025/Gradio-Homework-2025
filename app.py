import random
import gradio as gr
import json
import requests
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 聊天功能配置
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
SILICONFLOW_API_KEY = "sk-rkdqjxiipwxzxhqpjbovjmbxymagvunuyhqnlzsycojanaqi"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

def LLM_response(message, history):
    """流式生成响应的生成器函数"""
    # 构建消息历史
    messages = []
    for human, assistant in history:
        if (human):
            messages.append({"role": "user", "content": human})
        if (assistant):
            messages.append({"role": "assistant", "content": assistant})
    #messages.append({"role": "user", "content": message})

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
    logging.info("LLM已转发提交")
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
                        
                        # 解析JSON
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
with gr.Blocks(title="智能助手应用") as demo:
    # 状态变量
    username = gr.State("admin")
    current_page = gr.State(0)
    chat_history = gr.State([])
    
    # 整个应用布局
    with gr.Row():
        # 左侧导航栏
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("### 导航菜单")
            home_btn = gr.Button("首页", size="sm")
            feature1_btn = gr.Button("自动批注", size="sm")
            feature2_btn = gr.Button("自动摘抄", size="sm")
            yan_btn = gr.Button("关于严小希", size="sm")
            settings_btn = gr.Button("设置", size="sm")
            
            gr.Markdown("---")
            gr.Markdown(f"用户: admin")
            logout_btn = gr.Button("退出登录", size="sm")
        
        # 右侧内容区
        content_area = gr.Column()
        
        # 页面容器
        with gr.Column(visible=True) as home_container:
            # 聊天界面
            gr.Markdown("## 智能助手聊天界面")
            gr.Markdown("与AI助手交流，提问问题或进行对话")
            
            chatbot = gr.Chatbot(height=500, label="AI助手")
            msg = gr.Textbox(label="请输入消息", placeholder="输入您的问题后按Enter发送...")
            clear_btn = gr.Button("清空对话")
            file_upload = gr.File(label="上传文件（支持txt, pdf, docx）")
            file_output = gr.Markdown()
            
            # 聊天响应函数
            def respond(message, chat_history):
                # 立即添加用户消息（右侧显示用户输入）
                chat_history.append((message, ""))
                yield "", chat_history
                
                full_response = ""
                # 关键：传递完整的历史记录（包括刚添加的用户消息）
                for chunk in LLM_response(message, chat_history):
                    full_response = chunk
                    # 更新最后一条助手的回复内容（不创建新记录）
                    chat_history[-1] = (message, full_response)
                    yield "", chat_history
        
        # 自动批注页面
        with gr.Column(visible=False) as feature1_container:
            gr.Markdown("## 自动批注功能")
            gr.Markdown("上传文本或文档，系统会自动添加批注")
            
            with gr.Row():
                text_input = gr.Textbox(label="输入文本", lines=10, placeholder="在此输入要批注的文本...")
                file_input = gr.File(label="或上传文档", file_types=[".txt", ".pdf", ".docx"])
            
            generate_btn = gr.Button("开始批注")
            output_area = gr.Textbox(label="批注结果", interactive=False, lines=10)
        
        # 自动摘抄页面
        with gr.Column(visible=False) as feature2_container:
            gr.Markdown("## 自动摘抄功能")
            gr.Markdown("上传文档，系统会自动提取关键内容和摘要")
            
            file_input_f2 = gr.File(label="上传文档", file_types=[".pdf", ".docx", ".txt"])
            extract_btn = gr.Button("开始提取")
            
            with gr.Row():
                key_points = gr.Textbox(label="关键要点", interactive=False, lines=8)
                summary = gr.Textbox(label="文章摘要", interactive=False, lines=8)
        
        # 关于严小希页面
        with gr.Column(visible=False) as yan_container:
            gr.HTML(yan_page_html())
        
        # 设置页面
        with gr.Column(visible=False) as settings_container:
            gr.Markdown("## 系统设置")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 界面设置")
                    dark_mode = gr.Checkbox(label="深色模式")
                    notification = gr.Checkbox(label="启用通知")
                
                with gr.Column():
                    gr.Markdown("### AI设置")
                    ai_temperature = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.7, label="创造力")
                    ai_max_tokens = gr.Slider(minimum=100, maximum=2000, step=100, value=1024, label="最大生成长度")
            
            save_btn = gr.Button("保存设置")
            status = gr.Markdown("")
    
    # 页面切换函数
    def show_page(page_idx):
        return [gr.update(visible=page_idx==i) for i in range(5)]
    
    # 聊天页面的事件
    msg.submit(
        respond,
        inputs=[msg, chat_history],
        outputs=[msg, chatbot]
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
    logout_btn.click(
        lambda: [-1, "请登录", *[gr.update(visible=False) for _ in range(5)]],
        outputs=[current_page, content_area, home_container, feature1_container, feature2_container, yan_container, settings_container]
    )
    
    # 初始加载显示首页
    demo.load(lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)],
              outputs=[home_container, feature1_container, feature2_container, yan_container, settings_container])

# 启动应用
if __name__ == "__main__":
    demo.launch()