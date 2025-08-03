# Python语言与人工智能应用大作业

## 项目简介
这是一个集成了多种AI功能的交互式Web应用，包含聊天对话、自动摘抄、街景定位等功能模块。本项目是北京大学2025暑期Python语言与人工智能应用课程的期末大作业。

## 功能特性

### 智能聊天

• 与虚拟助手"严小希"进行自然语言对话

• 支持语音输出功能

• 上下文感知的对话体验

### 自动摘抄

• 从文本提取关键内容（支持图片OCR）

• 智能分类和组织摘抄内容

• 支持导出为可下载文件

• 结合手写字体生成可快速完成摘抄/批注等任务

### 街景定位

• 上传街景图片进行地理位置分析

• 识别图片中的电话号码、车牌号等线索

• 显示地图定位结果和AI分析过程

• 通过图片识别所在地/帮助[图寻](https://tuxun.fun/)的小工具

### 其他功能

• 关于严小希的个人介绍页面

• 响应式界面设计

• 用户友好的交互体验

## 技术栈

• 前端框架: Gradio

• AI服务: 硅基流动API

• OCR引擎: PaddleOCR

• 地图服务: 腾讯地图API

• 语音合成: CosyVoice2模型

## 安装与运行

### 环境要求

• Python 3.8+

• pip 20+

### 安装步骤

#### 克隆仓库
```
git clone https://github.com/xiaoce-2025/Gradio-Homework-2025.git
```

#### 安装依赖
```
pip install -r requirements.txt
```

#### 运行程序测试环境
```
python app.py
```

#### 配置API密钥
运行程序后，打开生成的config.ini文件，将其中的[TextModel]api_key项替换为你自己的硅基流动API密钥。

[获取硅基流动API↗](https://cloud.siliconflow.cn/me/account/ak)

#### 运行应用
```
\\ 开发者模式
gradio app.py

\\ 运行模式
python app.py
```

#### 访问应用：
打开浏览器访问 http://localhost:7860

## 使用说明

### 聊天功能：
   • 在输入框中输入消息，按Enter发送

   • 点击"语音输出"按钮收听AI回复

   • 点击"清空对话"重置聊天记录

### 自动摘抄：
   • 上传图片或粘贴文本内容

   • 点击"摘抄，启动！"按钮开始处理

   • 结果可导出为文本文件

### 街景定位：
   • 上传街景图片

   • 点击"开始定位"按钮

   • 查看地图定位结果和AI分析过程

## 项目结构

Yanxx-Workspace/
├── app.py                  # 主应用程序
├── ConfigManager.py        # 配置管理
├── YanxxDialog.py          # 对话逻辑
├── TextActracts.py         # 文本摘抄功能
├── grolocate.py            # 地理位置功能
├── YanxxPage.py            # 关于页面内容
├── requirements.txt        # 依赖列表
└── README.md               # 项目说明


## 贡献指南

欢迎提交Issue和Pull Request！请遵循以下准则：

• 提交前运行代码格式化工具

• 为新增功能添加测试用例

• 更新相关文档