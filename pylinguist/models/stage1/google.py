# pylinguist/models/stage1/google.py
from .base import BaseTranslator
from ...utils.logger import setup_logger
from deep_translator import GoogleTranslator as GTranslator

logger = setup_logger('pylinguist.stage1.google')

class GoogleTranslator(BaseTranslator):
    def __init__(self, source_lang: str, target_lang: str):
        super().__init__(source_lang, target_lang)
        self.translator = GTranslator(source=source_lang, target=target_lang)
        self.char_limit = 5000
        
    def translate_text(self, text: str) -> str:
        if not text or text.isspace():
            return text
            
        try:
            text = text[:self.char_limit] if len(text) > self.char_limit else text
            return self.translator.translate(text)
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            return text