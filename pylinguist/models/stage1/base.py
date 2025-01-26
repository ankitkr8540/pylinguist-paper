# base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import re
from ...utils.logger import setup_logger


logger = setup_logger('pylinguist.stage1.base')

class BaseTranslator(ABC):
    def __init__(self, source_lang: str, target_lang: str):
        self.source_lang = source_lang
        self.target_lang = target_lang
        
    def translate_code(self, code: str) -> str:
        try:
            lines = code.split('\n')
            translated_lines = []
            
            def is_english_word(token: str) -> bool:
                return all(c.isascii() and (c.isalpha() or c.isdigit() or c == '_') for c in token)
            
            def process_code_token(text: str) -> str:
                translated = self.translate_text(text)
                if translated and ' ' in translated:
                    return '_'.join(translated.split())
                return translated
            
            for line in lines:
                indent = len(line) - len(line.lstrip())
                line = line.lstrip()
                
                code_part, *comment = line.split('#', 1) + ['']
                tokens = re.findall(r'\w+(?:_\w+)*\(|\w+(?:_\w+)*|\s+|[^\w\s]|"[^"]*"|\'[^\']*\'', code_part)
                translated_tokens = []
                
                for token in tokens:
                    # Preserve string literals
                    if token.startswith(("'", '"')):
                        translated_tokens.append(token)
                        continue
                        
                    if token.isspace() or token in '()[]{},:+-*/=%<>!':
                        translated_tokens.append(token)
                        continue
                    
                    is_function = token.endswith('(')
                    base_token = token[:-1] if is_function else token
                    
                    if '_' in base_token:
                        parts = base_token.split('_')
                        if all(is_english_word(part) for part in parts):
                            phrase = base_token.replace('_', ' ')
                            translated = process_code_token(phrase)
                            if translated:
                                if is_function:
                                    translated += '('
                                translated_tokens.append(translated)
                                continue
                        translated_tokens.append(token)
                        continue
                    
                    if is_english_word(base_token):
                        translated = process_code_token(base_token)
                        if translated:
                            if is_function:
                                translated += '('
                            translated_tokens.append(translated)
                            continue
                    
                    translated_tokens.append(token)
                
                translated_line = ''.join(translated_tokens)
                if comment[0]:
                    translated_comment = self.translate_text(comment[0].strip())
                    translated_line += f" # {translated_comment}"
                
                translated_lines.append(' ' * indent + translated_line)
                
            return '\n'.join(translated_lines)
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return code


    @abstractmethod
    def translate_text(self, text: str) -> str:
        pass
