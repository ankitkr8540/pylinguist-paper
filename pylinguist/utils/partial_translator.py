import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import sys

def setup_logger():
    # Logger setup implementation here
    pass

logger = setup_logger()

class PartialTranslator:
    def __init__(self, source_lang: str, target_lang: str, 
                 keywords_path: Path = Path("data/keywords/Joshua_Keywords.csv")):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.keywords_path = keywords_path
        self.keyword_dict = self._load_keywords()

    def _load_keywords(self) -> Dict[str, str]:
        try:
            keywords_df = pd.read_csv(self.keywords_path)
            source_col = f"{self.source_lang}Key.txt"
            target_col = f"{self.target_lang}Key.txt"
            
            translation_dict = {
                str(row[source_col]).strip(): str(row[target_col]).strip()
                for _, row in keywords_df.iterrows()
                if pd.notna(row[source_col]) and pd.notna(row[target_col])
            }
            
            logger.info(f"Loaded {len(translation_dict)} keywords")
            return translation_dict
            
        except Exception as e:
            logger.error(f"Keywords loading failed: {str(e)}")
            sys.exit(1)

    def _extract_components(self, line: str) -> Dict[str, str]:
        """Extract code, comments and strings from line."""
        components = {'code': line, 'comment': '', 'strings': []}
        
        # Extract comments
        if '#' in line:
            components['code'], components['comment'] = line.split('#', 1)
            components['comment'] = '#' + components['comment']
            
        # Extract string literals
        string_pattern = r'(\".*?\"|\'.*?\')'
        components['strings'] = re.findall(string_pattern, components['code'])
        components['code'] = re.sub(string_pattern, 'STRING_PLACEHOLDER', components['code'])
        
        return components

    def _translate_token(self, token: str) -> str:
        """Translate individual token with special handling."""
        if token.isspace():
            return token
            
        # Handle function calls with parentheses
        if token in ['print', 'input', 'len', 'range', 'str', 'int', 'float']:
            return self.keyword_dict.get(token, token) + ' '
            
        # Handle compound words
        if '_' in token:
            parts = token.split('_')
            translated_parts = [self.keyword_dict.get(part, part) for part in parts]
            return '_'.join(translated_parts)
            
        return self.keyword_dict.get(token, token)

    def _translate_code_part(self, code: str) -> str:
        """Translate code while preserving structure."""
        tokens = re.findall(r'[a-zA-Z_]+|\d+|[^\w\s]|\s+', code)
        translated_tokens = [self._translate_token(token) for token in tokens]
        return ''.join(translated_tokens)

    def translate_line(self, line: str) -> str:
        """Translate a single line of code."""
        if not line.strip():
            return line
            
        # Get indentation
        indent = len(line) - len(line.lstrip())
        components = self._extract_components(line)
        
        # Translate code part
        translated_code = self._translate_code_part(components['code'])
        
        # Restore strings
        for string in components['strings']:
            translated_code = translated_code.replace('STRING_PLACEHOLDER', string, 1)
            
        # Add back comment and indentation
        return ' ' * indent + translated_code + components['comment']

    def translate_code(self, code: str) -> str:
        """Translate entire code snippet."""
        if not isinstance(code, str):
            return ""
            
        if '\\n' in code:
            lines = code.strip("'\"").split('\\n')
            translated_lines = [self.translate_line(line.strip()) for line in lines]
            return '\\n '.join(translated_lines)
            
        lines = code.split('\n')
        return '\n'.join(self.translate_line(line) for line in lines)

def partial_translate_examples(data_path: Path, source_lang: str, target_lang: str, 
                            start_index: int, test_samples: int, train_samples: int) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        total_samples = test_samples + train_samples
        if start_index + total_samples > len(df):
            start_index = 0
        
        selected_df = df.iloc[start_index:start_index + total_samples]
        translator = PartialTranslator(source_lang, target_lang)
        
        translations = [
            {
                'English_code': row['English_code'],
                'Partial_translated_code': translator.translate_code(row['English_code'])
            }
            for _, row in tqdm(selected_df.iterrows(), total=len(selected_df), desc="Translating code")
        ]
        
        return pd.DataFrame(translations)
        
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        raise