import json
import requests
import logging
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
                                    #logging.info(f"收到思考内容{content}")
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
                                    #logging.info(f"收到回复内容{content}")
                                    yield accumulated_text
                        except Exception as e:
                            # 解析错误时继续处理下一个chunk
                            continue
    except Exception as e:
        yield f"请求失败: {str(e)}"
