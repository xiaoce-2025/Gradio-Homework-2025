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