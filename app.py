import random
import gradio as gr
"""
用户输入后的回调函数 random_response
参数：
message: 用户此次输入的消息
history: 历史聊天记录，比如 [["use input 1", "assistant output 1"],["user input 2", "assistant output 2"]]
​
返回值：输出的内容
"""
def random_response(message, history):
    return random.choice(["Yes", "No"])
            
gr.ChatInterface(random_response).launch()