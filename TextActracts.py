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




# 初始化OCR引擎（使用中英文模型）
ocr = PaddleOCR(use_angle_cls=True, lang="ch", enable_mkldnn=False)

class TextOCR:
    @staticmethod
    def preprocess_image(img):
        # 灰度化
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 二值化
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 降噪
        denoised = cv2.fastNlMeansDenoising(thresh, h=10)
        return denoised

    @staticmethod
    def ocr_processing(file_path):
        logging.debug(f"收到OCR提交:{file_path}")
        try:
            # 读取并处理图片
            img = cv2.imread(file_path)
            img = TextOCR.preprocess_image(img)
            
            # OCR识别
            result = ocr.ocr(img, cls=True)
            texts = [line[1][0] for line in result[0]] if result else []
            
            return '\n'.join(texts)

        except Exception as e:
            raise Exception(f"OCR处理失败: {str(e)}")

class AutoExtract:
    @staticmethod
    def AI_Auto_Extract(config, text):
        # 模型关键信息检查
        if not (config['apiKey'] and config['apiUrl'] and config['ModelName']):
            return []

        # 模型温度参数在0-2
        temperature = max(0, min(2, config.get('ModelTemperature', 1)))
        
        # 模型maxtoken>1且为整数
        max_token = int(config.get('maxToken', 4096))
        max_token = max(1, min(max_token, 4096))

        client = OpenAI(
            api_key=config['apiKey'],
            base_url=config['apiUrl'],
        )

        system_prompt = """
        用户将提供一段文本，请对这段文本进行摘抄，并对摘抄内容进行点评。摘抄你所有觉得比较好的内容，点评字数合适即可，按顺序输出原文文本和点评。总句数不超过20条。并以JSON格式输出。

        示例输入：
        老人与海节选：大马林鱼开始快速地围着小渔船游动，将缆绳缠绕到了桅杆上，老人右手高举着钢叉，在它跃出水面的一瞬间，竭尽全力地向它的心脏掷去，一声哀鸣结束了大鱼的生命，它静静地浮在水面上……老人和大鱼一直相持到日落，双方已搏斗了两天一夜，老头不禁回想起年轻时在卡萨兰卡跟一个黑人比赛扳手的经历。一个鱼食送下四十英寸的深处，第二个鱼食送下七十五英寸的深处，第三个和第四个鱼使分别送到了大海下面一百英寸和一百二十五英寸的地方去了。一个孤独的老人拖着疲惫不堪的身子漂泊在茫茫的海面上活像个大战后的勇士。为了治服那条庞大的马林鱼，他已经费下了自己近乎所有的力气。
        
        示例JSON输出:
        {
            "source1":{
                "text": "大马林鱼开始快速地围着小渔船游动，将缆绳缠绕到了桅杆上，老人右手高举着钢叉，在它跃出水面的一瞬间，竭尽全力地向它的心脏掷去，一声哀鸣结束了大鱼的生命，它静静地浮在水面上……"
                "comment": "我的心也像一块大石头落了地。我非常钦佩老人那种毫不畏惧、坚持不懈的精神，虽然知道对手实力很强，但他没有丝毫退缩，而是迎难而上。正因为有了这种精神，老渔夫才获得了这场生死较量的胜利。我们在生活中也要学习老渔夫的精神，做事情不怕困难，才能取得成功。"
            }
            "source2": {
                "text": "一个鱼食送下四十英寸的深处，第二个鱼食送下七十五英寸的深处，第三个和第四个鱼使分别送到了大海下面一百英寸和一百二十五英寸的地方去了。"
                "comment": "《老人与海》产生的视觉形象，画面感很强，这与作者应用部分电影化手法是分不开的。作品一开始就使用了特写镜头，对帆和老人的面部做了展示。近景在老人下鱼食的细节上体现最为充分：一个一个放钓丝的动作那么仔细、真切。"
            }
        }
        """

        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}]

        logging.debug(f"已转发摘抄提交请求:{messages}")

        response = client.chat.completions.create(
            model=config['ModelName'],
            messages=messages,
            max_tokens=max_token,
            temperature=temperature,
            response_format={
                'type': 'json_object'
            }
        )

        logging.info(f"已收到摘抄返回结果{response}")
        
        # 解析JSON输出
        response_content = response.choices[0].message.content
        # 去除可能的代码块标记
        if response_content.startswith('```json'):
            response_content = response_content.replace('```json', '').replace('```', '').strip()
        elif response_content.startswith('```'):
            response_content = response_content.replace('```', '').strip()
        
        try:
            parsed_response = json.loads(response_content)
        except json.JSONDecodeError as e:
            logging.error(f"JSON解析失败: {e}, 原始响应: {response_content}")
            raise Exception(f"JSON解析失败: {str(e)}")
        
        # 格式标准化
        result = []
        for key, content in parsed_response.items():
            if 'text' in content and 'comment' in content:
                result.append({
                    "sentence": content['text'], 
                    "comment": content['comment']
                })
        return result


def process_ocr(file):
    try:
        if file is None:
            return "请先选择图片文件！"
        
        # 中文路径兼容性处理
        if isinstance(file, str):
            # 处理Gradio返回的Windows中文路径
            try:
                # 尝试直接打开中文路径
                with open(file, "rb") as f:
                    content = f.read()
            except UnicodeEncodeError:
                # 使用UTF-8编码处理路径
                with open(file.encode('utf-8'), "rb") as f:
                    content = f.read()
        else:
            # 从文件对象读取内容
            content = file.read() if hasattr(file, "read") else file
        
        # 创建临时文件（自动处理中文）
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            # 确保使用正确的二进制写入
            tmp_file.write(content)
            file_path = tmp_file.name
        
        # OCR处理
        ocr_text = TextOCR.ocr_processing(file_path)
        
        # 清理临时文件
        try:
            os.unlink(file_path)
        except Exception as e:
            logging.warning(f"临时文件删除失败: {str(e)}")
        
        return ocr_text
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        logging.error(f"OCR处理错误: {str(e)}\n{error_detail}")
        return f"OCR处理错误: {str(e)}"
        
def extract_excerpts(text, config):
    if not text:
        return [], "请输入需要摘抄的文本！"
    
    try:
        excerpts = AutoExtract.AI_Auto_Extract(config, text)
        
        # 生成HTML展示
        html_content = "<div class='excerpts-list'>"
        for idx, excerpt in enumerate(excerpts):
            html_content += f"""
            <div class='excerpt-item' style='margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);'>
                <div class='sentence-box' style='display: flex; align-items: center; margin-bottom: 10px;'>
                    <span class='sentence-num' style='background: #2196F3; color: white; padding: 2px 8px; border-radius: 4px; margin-right: 10px;'>#{idx+1}</span>
                    <p class='sentence-text' style='margin: 0;'>{excerpt['sentence']}</p>
                </div>
                <textarea class='comment-input' style='width: 95%; padding: 10px; border: 1px solid #ddd; border-radius: 6px; min-height: 80px; resize: vertical;'>{excerpt['comment']}</textarea>
            </div>
            """
        html_content += "</div>"
        
        return html_content, excerpts
    except Exception as e:
        return f"<div class='error'>摘抄处理失败: {str(e)}</div>", []
    
def export_excerpts(excerpts, format_choice="TXT"):
    if not excerpts:
        return "没有摘抄内容可导出！", None
    
    # 生成TXT内容
    def generate_txt_content():
        content = "摘抄内容\n"
        content += "="*50 + "\n\n"
        for idx, excerpt in enumerate(excerpts):
            content += f"{idx+1}. {excerpt['sentence']}\n"
            if excerpt.get('comment'):
                content += f"   点评：{excerpt['comment']}\n"
            content += "\n"
        return content
    
    # 生成Markdown内容
    def generate_markdown_content():
        content = "# 摘抄内容\n\n"
        for idx, excerpt in enumerate(excerpts):
            content += f"## {idx+1}. {excerpt['sentence']}\n\n"
            if excerpt.get('comment'):
                content += f"**点评：** {excerpt['comment']}\n\n"
            content += "---\n\n"
        return content
    
    # 生成Word内容（HTML格式）
    def generate_word_content():
        content = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>摘抄内容</title>
            <style>
                body { font-family: Arial, sans-serif; }
                .excerpt { margin-bottom: 20px; }
                .sentence { font-weight: bold; }
                .comment { margin-left: 20px; }
            </style>
        </head>
        <body>
            <h1>摘抄内容</h1>
        """
        
        for idx, excerpt in enumerate(excerpts):
            content += f"""
            <div class="excerpt">
                <p class="sentence">{idx+1}. {excerpt['sentence']}</p>
                <p class="comment">{excerpt.get('comment', '')}</p>
            </div>
            """
        
        content += "</body></html>"
        return content
    
    # 根据选择生成内容
    if format_choice == "TXT":
        content = generate_txt_content()
        filename = "excerpts.txt"
    elif format_choice == "Markdown":
        content = generate_markdown_content()
        filename = "excerpts.md"
    elif format_choice == "Word":
        content = generate_word_content()
        filename = "excerpts.html"
    
    # 创建临时文件
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    # 返回状态消息和文件对象（Gradio会自动处理下载）
    return f"导出成功！点击下方链接下载", file_path