# pylinguist/utils/partial_translator.py

import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Optional
import logging
from ..utils.language_extractor import extract_keyword_header, extract_language
from ..utils.logger import setup_logger
from tqdm import tqdm
import sys

logger = setup_logger()

class PartialTranslator:
    """Handles partial translation using Joshua keywords while preserving code structure."""
    
    def __init__(self, source_lang: str, target_lang: str, 
                 keywords_path: Path = Path("data/keywords/Joshua_Keywords.csv")):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.keywords_path = keywords_path
        self.keyword_dict = self._load_keywords()
        
    def _load_keywords(self) -> Dict[str, str]:
        """Load keyword mappings for specified languages."""
        try:
            source_col = extract_keyword_header(self.source_lang)
            target_col = extract_keyword_header(self.target_lang)
            
            keywords_df = pd.read_csv(self.keywords_path)
            
            translation_dict = {}
            for _, row in keywords_df.iterrows():
                source_word = str(row[source_col]).strip()
                target_word = str(row[target_col]).strip()
                if pd.notna(source_word) and pd.notna(target_word):
                    translation_dict[source_word] = target_word
                    
            logger.info(f"Loaded keywords dictionary for {extract_language(self.source_lang)} "
                     f"to {extract_language(self.target_lang)}")
                     
            return translation_dict
            
        except Exception as e:
            logger.error(f"Error loading keywords: {str(e)}")
            sys.exit(1)
            raise
    
    def translate_word(self, word: str) -> str:
        """Translate single word unless it contains underscores."""
        return word if '_' in word else self.keyword_dict.get(word, word)

    def translate_code(self, code: str) -> str:
        lines = code.split('\n')
        translated_lines = []
        
        function_pattern = r'(\w+(?:_\w+)*)(\s*\()([^)]*?)(\))'
        
        for line in lines:
            if not line.strip():
                translated_lines.append(line)
                continue
                
            code_part, comment_part = line, ""
            if '#' in line:
                code_part, comment_part = line.split('#', 1)
                comment_part = '#' + comment_part
            
            strings = []
            string_pattern = r'(\".*?\"|\'.*?\')'
            for match in re.finditer(string_pattern, code_part):
                strings.append(match.group(0))
            
            placeholder_code = re.sub(string_pattern, 'STRING_PLACEHOLDER', code_part)
            
            def tokenize_func(match):
                func_name = match.group(1)
                translated_func = self.translate_word(func_name)
                params = match.group(3)
                translated_params = []
                
                if params:
                    param_tokens = re.findall(r'\w+(?:_\w+)*|[^\w\s]', params)
                    translated_params = [self.translate_word(token) for token in param_tokens]
                    
                return f"{translated_func}{match.group(2)}{' '.join(translated_params)}{match.group(4)}"
                
            placeholder_code = re.sub(function_pattern, tokenize_func, placeholder_code)
            
            tokens = re.findall(r'\w+(?:_\w+)*|\s+|[^\w\s]', placeholder_code)
            translated_tokens = [self.translate_word(token) if not token.isspace() else token 
                                for token in tokens]
            
            translated_code = ''.join(translated_tokens)
            for string in strings:
                translated_code = translated_code.replace('STRING_PLACEHOLDER', string, 1)
                
            translated_lines.append(translated_code + comment_part)
        
        return '\n'.join(translated_lines)
    
    
def partial_translate_examples(data_path: Path, source_lang: str, target_lang: str, 
                            start_index: int, stage1_samples: int, stage2_samples: int) -> pd.DataFrame:
    """
    Translate multiple examples from dataset.
    Returns DataFrame with original and translated code.
    """
    try:
        df = pd.read_csv(data_path)
        translator = PartialTranslator(source_lang, target_lang)
        
        total_samples = stage1_samples + stage2_samples
        if start_index + total_samples > len(df):
            logger.warning("Requested range exceeds dataset size. Adjusting start index...")
            start_index = 0
            
        selected_df = df.iloc[start_index:start_index + total_samples]
        
        translations = []
        
        for _, row in tqdm(pd.DataFrame(selected_df).iterrows(), total=len(selected_df), desc="Translating code"):
            translated_code = translator.translate_code(row['English_code'])
            translations.append({
            'English_code': row['English_code'],
            'Partial_translated_code': translated_code
            })
        
        return pd.DataFrame(translations)
        
    except Exception as e:
        logger.error(f"Error in partial translation: {str(e)}")
        raise