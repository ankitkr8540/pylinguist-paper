from openai import OpenAI
import os
import time
from typing import Optional
from dotenv import load_dotenv
from pylinguist.utils.language_extractor import extract_language
from .base import BaseEnhancer
from ...utils.logger import setup_logger
import pandas as pd

logger = setup_logger('stage2.gpt')
load_dotenv()
class DeepSeekEnhancer(BaseEnhancer):
    """GPT model implementation for Stage 2 translation enhancement."""
    
    def __init__(self, source_lang: str, target_lang: str, translator_name: str):

        target_lang = extract_language(target_lang)
        source_lang = extract_language(source_lang)
        super().__init__(source_lang, target_lang, translator_name)
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment")
            
        self.client = OpenAI(api_key= self.api_key, base_url="https://openrouter.ai/api/v1")

        self.model = "deepseek/deepseek-r1:free"
        self.temperature = 0
        self.max_retries = 3
        self.retry_delay = 1
        
        
    def _clean_response(self, response: str) -> str:
        """Remove code block markers from response."""
        if not response:
            return response
            
        response = response.strip()
        if "```python" in response:
            response = response.split("```python")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].strip()
        return response

    def enhance_translation(self, code: str, examples_df: pd.DataFrame) -> str:
        """Enhance translation using GPT."""
        # Check cache first
        cache_key = f"{code}_{len(examples_df)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        try:
            prompt = self.create_prompt(examples_df, code)
            
            for attempt in range(self.max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": f"You are a Expert Python code translator who understands the nuanses of language in coding and converts code from {self.source_lang} to  {self.target_lang} code while preserving functionality. Return only the translated code without any explanation."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=self.temperature
                    )
                    
                    translated = self._clean_response(response.choices[0].message.content)
                    self.cache[cache_key] = translated
                    return translated
                    
                except Exception as e:
                    logger.warning(f"Translation attempt {attempt + 1} failed: {str(e)}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    continue
                    
            logger.error("All translation attempts failed")
            return code
            
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return code