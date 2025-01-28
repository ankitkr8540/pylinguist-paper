from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd
from ...utils.logger import setup_logger

logger = setup_logger('stage2.base')

class BaseEnhancer(ABC):
    """Base class for Stage 2 translation enhancement."""
    
    def __init__(self, source_lang: str, target_lang: str, translator_name: str):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.translator_name = translator_name
        self.cache = {}
        
    def create_prompt(self, examples_df: pd.DataFrame, code_to_translate: str) -> str:
        """Create prompt with examples and code to translate."""
        examples_text = ""
        for i, row in examples_df.iterrows():
            examples_text += f"\nExample {i+1}:\n"
            examples_text += f"Original code:\n{row['English_code']}\n"
            examples_text += f"Complete translation:\n{row[f'{self.translator_name}_translated_code']}\n"
            examples_text += "-" * 50 + "\n"
            
        prompt = f"""Complete the translation of this partially {self.source_lang} Python code to completely {self.target_lang} python code:
            - Translate variable names, function names, strings and comments to {self.target_lang}
            - Join multi-word {self.target_lang} translations with underscores
            - Break down compound {self.source_lang} words separated by underscores and translate each part into sensible {self.target_lang} and join them back with underscores
            - Preserve code structure and syntax
            - Here are some examples of translations:

            Examples:
            {examples_text}

            Code to translate:
            {code_to_translate}

            Return only the complete translated code."""
        
        return prompt
        
    @abstractmethod
    def enhance_translation(self, code: str, examples_df: pd.DataFrame) -> str:
        """Enhance translation using model."""
        pass