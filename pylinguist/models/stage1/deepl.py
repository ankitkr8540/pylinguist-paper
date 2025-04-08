# pylinguist/models/stage1/deepl.py

import os
import time
import requests
from typing import Dict, List, Optional
from .base import BaseTranslator
from ...utils.logger import setup_logger

logger = setup_logger('pylinguist.stage1.deepl')

class DeepLTranslator(BaseTranslator):
    LANG_MAP = {'en': 'EN', 'de': 'DE', 'fr': 'FR', 'es': 'ES', 'it': 'IT',
                'ja': 'JA', 'hi': 'HI', 'zh-CN': 'ZH', 'pt': 'PT', 'ru': 'RU'}
    
    def __init__(self, source_lang: str, target_lang: str, api_key: Optional[str] = None):
        super().__init__(source_lang, target_lang)
        self.api_key = api_key or os.getenv('DEEPL_API_KEY')
        self.base_url = "https://api-free.deepl.com/v1"
        self.char_limit = 30000
        
    def translate_text(self, text: str) -> str:
        if not text or text.isspace():
            return text
            
        try:
            text = text[:self.char_limit] if len(text) > self.char_limit else text
            response = requests.post(
                f"{self.base_url}/translate",
                headers={'Authorization': f'DeepL-Auth-Key {self.api_key}'},
                data={
                    'text': text,
                    'source_lang': self.LANG_MAP[self.source_lang.lower()],
                    'target_lang': self.LANG_MAP[self.target_lang.lower()],
                    'preserve_formatting': True
                }
            )
            response.raise_for_status()
            return response.json()['translations'][0]['text']
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            return text