import configparser
import os

class ConfigManager:
    def __init__(self, config_path="config.ini"):
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self.load_config()
    
    def load_config(self):
        """加载配置文件"""
        if os.path.exists(self.config_path):
            self.config.read(self.config_path)
        else:
            # 创建默认配置
            self.config['TextModel'] = {
                'name': 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
                'api_url': 'https://api.siliconflow.cn/v1/',
                'api_key': 'your-api-key-here'
            }
            self.config['TextModelParams'] = {
                'max_tokens': '4000',
                'temperature': '0.7'
            }
            self.save_config()
    
    def save_config(self):
        """保存配置到文件"""
        with open(self.config_path, 'w') as configfile:
            self.config.write(configfile)
    
    def get_text_model_config(self):
        """获取文本模型配置"""
        return {
            'name': self.config.get('TextModel', 'name', fallback='gpt-3.5-turbo'),
            'api_url': self.config.get('TextModel', 'api_url', fallback=''),
            'api_key': self.config.get('TextModel', 'api_key', fallback=''),
            'max_tokens': self.config.getint('TextModelParams', 'max_tokens', fallback=2000),
            'temperature': self.config.getfloat('TextModelParams', 'temperature', fallback=0.7)
        }