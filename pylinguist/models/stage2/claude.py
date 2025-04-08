# pylinguist/models/stage2/claude.py
import anthropic
import os
import time
from typing import Optional
from .base import BaseEnhancer
import pandas as pd
from dotenv import load_dotenv
from ...utils.logger import setup_logger

logger = setup_logger('stage2.claude')
load_dotenv()
class ClaudeTranslator(BaseEnhancer):
    """Claude model implementation for Stage 2 translation enhancement."""
    
    def __init__(self, source_lang: str, target_lang: str, translator_name: str):
        super().__init__(source_lang, target_lang, translator_name)
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-3-5-haiku-20241022"
        self.max_retries = 3
        self.retry_delay = 2
        
    def _clean_response(self, response: str) -> str:
        """Clean Claude's response output."""
        if "```python" in response:
            return response.split("```python")[1].split("```")[0].strip()
        return response.strip()

    def enhance_translation(self, code: str, examples_df: pd.DataFrame) -> str:
        cache_key = f"{code}_{len(examples_df)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        try:
            prompt = self.create_prompt(examples_df, code)
            
            for attempt in range(self.max_retries):
                try:
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=4000,
                        temperature=0,
                        system=f"You are a Expert Python code translator who understands the nuanses of language in coding and converts code from {self.source_lang} to  {self.target_lang} code while preserving functionality. Return only the translated code without any explanation.",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    translated = self._clean_response(response.content[0].text)
                    self.cache[cache_key] = translated
                    return translated
                except Exception as e:
                    logger.warning(f"Attempt {attempt+1} failed: {str(e)}")
                    time.sleep(self.retry_delay)
            return code
        except Exception as e:
            logger.error(f"Claude translation error: {str(e)}")
            return code