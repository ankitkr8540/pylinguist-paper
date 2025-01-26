# pylinguist/models/stage1/deepl.py

import os
import time
import requests
from typing import Dict, List, Optional
from .base import BaseTranslator, TranslationError
from ...utils.logger import setup_logger

logger = setup_logger('pylinguist.stage1.deepl')

class DeepLTranslator(BaseTranslator):
    """DeepL translation implementation for Stage 1."""
    
    # DeepL language code mappings
    LANGUAGE_MAPPINGS = {
        'en': 'EN',  # English
        'de': 'DE',  # German
        'fr': 'FR',  # French
        'es': 'ES',  # Spanish
        'it': 'IT',  # Italina
        'ja': 'JA',  # Japanese
        'hi': 'HI',  # Hindi
        'zh': 'ZH',  # Chinese
        'pt': 'PT',  # Portuguese
        'ru': 'RU',  # Russian
    }
    
    def __init__(self, source_lang: str, target_lang: str, retry_attempts: int = 3, 
                 api_key: Optional[str] = None):
        """
        Initialize DeepL translator.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            retry_attempts: Number of retry attempts
            api_key: DeepL API key (optional, will look in environment if not provided)
        """
        super().__init__(source_lang, target_lang, retry_attempts)
        self.service_name = "deepl"
        self.api_key = api_key or os.getenv('DEEPL_API_KEY')
        
        if not self.api_key:
            raise TranslationError("DeepL API key not provided or found in environment")
        
        self.base_url = "https://api-free.deepl.com/v2"
        self.char_limit = 30000  # DeepL's limit per request
        self.retry_delay = 1  # seconds
        self.monthly_char_count = 0
        self.max_monthly_chars = 500000  # Free tier limit
        
    def _map_language_code(self, code: str, is_source: bool = True) -> str:
        """Map language code to DeepL format."""
        mapped_code = self.LANGUAGE_MAPPINGS.get(code.lower())
        if not mapped_code:
            raise TranslationError(f"Unsupported language code: {code}")
        
        # Source languages might need additional mapping
        if is_source and mapped_code in ['EN', 'PT']:
            return f"{mapped_code}-{mapped_code}"
        return mapped_code
        
    def _check_usage_limits(self, text_length: int):
        """Check if we're within usage limits."""
        if self.monthly_char_count + text_length > self.max_monthly_chars:
            raise TranslationError("Monthly character limit reached")
            
    def _get_usage_stats(self) -> Dict:
        """Get current API usage statistics."""
        try:
            response = requests.get(
                f"{self.base_url}/usage",
                headers={'Authorization': f'DeepL-Auth-Key {self.api_key}'}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting usage stats: {str(e)}")
            return {}
            
    def translate_text(self, text: str) -> Optional[str]:
        """
        Translate text using DeepL API.
        
        Args:
            text: Text to translate
        """
        if not text or text.isspace():
            return text
            
        # Check length limits
        text_length = len(text)
        if text_length > self.char_limit:
            logger.warning(f"Text exceeds {self.char_limit} characters, truncating...")
            text = text[:self.char_limit]
            text_length = self.char_limit
            
        self._check_usage_limits(text_length)
        
        # Map language codes
        source_lang = self._map_language_code(self.source_lang, True)
        target_lang = self._map_language_code(self.target_lang, False)
        
        for attempt in range(self.retry_attempts):
            try:
                response = requests.post(
                    f"{self.base_url}/translate",
                    headers={'Authorization': f'DeepL-Auth-Key {self.api_key}'},
                    data={
                        'text': text,
                        'source_lang': source_lang,
                        'target_lang': target_lang,
                        'preserve_formatting': True,
                        'formality': 'more'  # More formal translations for code
                    }
                )
                
                response.raise_for_status()
                result = response.json()
                translated = result['translations'][0]['text']
                
                # Update character count
                self.monthly_char_count += text_length
                
                if self.validate_translation(text, translated):
                    return translated
                    
                logger.warning("Translation validation failed, retrying...")
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Translation attempt {attempt + 1} failed: {str(e)}")
                
                if response.status_code == 429:  # Rate limit
                    time.sleep(self.retry_delay * 5)  # Longer wait for rate limits
                elif response.status_code == 456:  # Quota exceeded
                    raise TranslationError("Monthly quota exceeded")
                    
            except Exception as e:
                logger.warning(f"Unexpected error in attempt {attempt + 1}: {str(e)}")
                
            if attempt < self.retry_attempts - 1:
                time.sleep(self.retry_delay * (attempt + 1))
                
        logger.error("All translation attempts failed")
        return None
        
    def validate_languages(self) -> bool:
        """Validate that the language pair is supported."""
        try:
            source_lang = self._map_language_code(self.source_lang, True)
            target_lang = self._map_language_code(self.target_lang, False)
            return True
        except TranslationError:
            return False
        except Exception as e:
            logger.error(f"Error validating languages: {str(e)}")
            return False
    
    def get_supported_languages(self) -> Dict[str, List[str]]:
        """Get supported languages from DeepL."""
        try:
            # Get source languages
            source_response = requests.get(
                f"{self.base_url}/source_languages",
                headers={'Authorization': f'DeepL-Auth-Key {self.api_key}'}
            )
            
            # Get target languages
            target_response = requests.get(
                f"{self.base_url}/target_languages",
                headers={'Authorization': f'DeepL-Auth-Key {self.api_key}'}
            )
            
            source_response.raise_for_status()
            target_response.raise_for_status()
            
            return {
                'source_langs': [lang['language'].lower() for lang in source_response.json()],
                'target_langs': [lang['language'].lower() for lang in target_response.json()]
            }
            
        except Exception as e:
            logger.error(f"Error getting supported languages: {str(e)}")
            return {'source_langs': [], 'target_langs': []}
    
    def get_service_info(self) -> Dict:
        """Get detailed service information."""
        info = super().get_service_info()
        usage_stats = self._get_usage_stats()
        
        info.update({
            'char_limit': self.char_limit,
            'retry_delay': self.retry_delay,
            'monthly_char_count': self.monthly_char_count,
            'max_monthly_chars': self.max_monthly_chars,
            'usage_stats': usage_stats
        })
        return info