import random
import gradio as gr
import json
import requests
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
"""
用户输入后的回调函数 random_response
参数：
message: 用户此次输入的消息
history: 历史聊天记录，比如 [["use input 1", "assistant output 1"],["user input 2", "assistant output 2"]]
​
返回值：输出的内容
"""
# 测试用配置
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"  # 替换为实际流式API地址
SILICONFLOW_API_KEY = "sk-rkdqjxiipwxzxhqpjbovjmbxymagvunuyhqnlzsycojanaqi"  # 替换为你的API密钥
MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"  # 替换为实际模型名称


def LLM_response(message, history):
    """
    流式生成响应的生成器函数
    """
    # 1. 构建对话历史
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})
    
    # 2. 创建请求头和数据
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1024,
        "stream": True  # 启用流式输出
    }
    
    # 3. 发送流式请求
    try:
        with requests.post(
            SILICONFLOW_API_URL,
            headers=headers,
            json=payload,
            stream=True  # 启用流式接收
        ) as response:
            if response.status_code != 200:
                yield f"API错误: {response.status_code} - {response.text}"
                return
                
            # 4. 处理流式响应
            accumulated_text = ""
            for chunk in response.iter_lines():
                # 过滤保持连接的空行
                if chunk:
                    # 解析事件流数据
                    decoded_chunk = chunk.decode('utf-8')
                    
                    # 检查是否是数据行
                    if decoded_chunk.startswith("data:"):
                        # 提取JSON部分
                        json_data = decoded_chunk[5:].strip()
                        
                        # 结束标记
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
            
demo = gr.ChatInterface(LLM_response)

if __name__ == "__main__":
    demo.launch()