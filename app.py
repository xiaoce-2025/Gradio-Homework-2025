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
        messages.append({"role": "user", "content": human})
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


# 定义页面内容
def blank_page():
    return "<div style='text-align:center; padding:50px;'><h2>空白页面</h2></div>"

def home_page():
    # 将组件转换为HTML
    return "<div style='text-align:center; padding:50px;'><h2>聊天页面</h2></div>"

def feature1_page():
    return """
    <div style="padding:20px;">
        <h2>自动批注功能</h2>
        <p>这是自动批注功能的示例页面</p>
        <textarea rows="10" style="width:100%" placeholder="请输入要批注的文本..."></textarea>
        <br>
        <button style="margin-top:10px;">开始批注</button>
    </div>
    """

def feature2_page():
    return """
    <div style="padding:20px;">
        <h2>自动摘抄功能</h2>
        <p>这是自动摘抄功能的示例页面</p>
        <input type="file" style="margin-bottom:10px;">
        <br>
        <button>提取摘抄</button>
    </div>
    """

from YanxxPage import Yanxx_Page
def yan_page():
    return Yanxx_Page()

def settings_page():
    return """
    <div style="padding:20px;">
        <h2>设置</h2>
        <label><input type="checkbox"> 启用通知</label><br>
        <label><input type="checkbox"> 深色模式</label><br>
        <label>主题: 
            <select>
                <option>默认</option>
                <option>蓝色</option>
                <option>绿色</option>
            </select>
        </label>
    </div>
    """

# 主应用逻辑
def main_app(username, action):
    if action == "logout":
        return None, "请登录"
    
    if isinstance(action, int):
        idx = action
    else:
        idx = 0
    
    pages = [home_page, feature1_page, feature2_page, yan_page, settings_page]
    return pages[min(idx, len(pages)-1)](), f"用户: {username}"

# 创建 Gradio 界面
with gr.Blocks(title="Gradio App") as demo:
    username_state = gr.State("admin")
    current_page = gr.State(0)
    
    with gr.Row():
        # 左侧导航栏
        with gr.Column(scale=1, min_width=150):
            nav_items = gr.Radio(
                choices=["首页", "自动批注", "自动摘抄", "关于严小希", "设置"],
                type="index",
                label="导航菜单",
                value="首页"
            )
            
            gr.Markdown("---")
            user_display = gr.Markdown("用户: admin")
            
            logout_btn = gr.Button("退出登录")
        
        # 右侧内容区
        with gr.Column(scale=4):
            content_area = gr.HTML()
    
    # 初始化显示首页
    demo.load(fn=home_page, outputs=content_area)
    
    # 事件处理
    nav_items.change(
        fn=lambda idx: idx,
        inputs=nav_items,
        outputs=current_page,
        queue=False
    ).then(
        lambda idx: [
            home_page,
            feature1_page,
            feature2_page,
            yan_page,
            settings_page
        ][min(idx, 4)](),
        inputs=current_page,
        outputs=content_area
    )
    
    logout_btn.click(
        fn=lambda: (None, -1, "请登录", blank_page()),
        outputs=[username_state, current_page, user_display, content_area],
        queue=False
    )

# 启动应用
if __name__ == "__main__":
    demo.launch()