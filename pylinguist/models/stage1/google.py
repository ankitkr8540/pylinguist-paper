# pylinguist/models/stage1/google.py

import time
from typing import Dict, List, Optional
from deep_translator import GoogleTranslator as GTrans
from .base import BaseTranslator, TranslationError
from ...utils.logger import setup_logger

logger = setup_logger('pylinguist.stage1.google')

class GoogleTranslator(BaseTranslator):
    """Google Translate implementation for Stage 1 translation."""
    
    def __init__(self, source_lang: str, target_lang: str, retry_attempts: int = 3):
        super().__init__(source_lang, target_lang, retry_attempts)
        self.service_name = "google"
        self.translator = self._initialize_translator()
        self.char_limit = 5000
        self.retry_delay = 1  # seconds
        
    def _initialize_translator(self) -> GTrans:
        """Initialize Google translator with error handling."""
        try:
            return GTrans(source=self.source_lang, target=self.target_lang)
        except Exception as e:
            raise TranslationError(f"Failed to initialize Google Translator: {str(e)}")
            
    def translate_text(self, text: str) -> Optional[str]:
        """
        Translate text using Google Translate with retry mechanism.
        
        Args:
            text: Text to translate
        """
        if not text or text.isspace():
            return text
            
        if len(text) > self.char_limit:
            logger.warning(f"Text exceeds {self.char_limit} characters, truncating...")
            text = text[:self.char_limit]
            
        for attempt in range(self.retry_attempts):
            try:
                translated = self.translator.translate(text)
                # if self.validate_translation(text, translated):
                return translated
                    
                logger.warning("Translation validation failed, retrying...")
                
            except Exception as e:
                logger.warning(f"Translation attempt {attempt + 1} failed: {str(e)}")
                
            if attempt < self.retry_attempts - 1:
                time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                
        logger.error("All translation attempts failed")
        return None
        
    def validate_languages(self) -> bool:
        """Validate that the language pair is supported."""
        try:
            supported = self.get_supported_languages()
            source_supported = self.source_lang in supported.get('source_langs', [])
            target_supported = self.target_lang in supported.get('target_langs', [])
            
            if not source_supported:
                logger.error(f"Source language '{self.source_lang}' not supported")
            if not target_supported:
                logger.error(f"Target language '{self.target_lang}' not supported")
                
            return source_supported and target_supported
            
        except Exception as e:
            logger.error(f"Error validating languages: {str(e)}")
            return False
    
    def get_supported_languages(self) -> Dict[str, List[str]]:
        """Get supported languages from Google Translate."""
        try:
            langs = self.translator.get_supported_languages(as_dict=True)
            return {
                'source_langs': list(langs.keys()),
                'target_langs': list(langs.keys())
            }
        except Exception as e:
            logger.error(f"Error getting supported languages: {str(e)}")
            return {'source_langs': [], 'target_langs': []}
            
    def get_service_info(self) -> Dict:
        """Get detailed service information."""
        info = super().get_service_info()
        info.update({
            'char_limit': self.char_limit,
            'retry_delay': self.retry_delay
        })
        return info