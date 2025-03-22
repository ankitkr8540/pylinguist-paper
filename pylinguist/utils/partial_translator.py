# pylinguist/utils/partial_translator.py

import pandas as pd
import re
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Set
from ..utils.language_extractor import extract_keyword_header, extract_language
from ..utils.logger import setup_logger
from tqdm import tqdm
import sys

logger = setup_logger()

class PartialTranslator:
    """
    Handles partial translation using Joshua keywords while preserving code structure.
    Supports multiple languages including their specific character sets.
    """
    
    # Language-specific character ranges
    LANGUAGE_CHARS = {
        'hi': ('\u0900', '\u097F'),  # Devanagari (Hindi)
        'bn': ('\u0980', '\u09FF'),  # Bengali
        'zh': ('\u4E00', '\u9FFF'),  # Chinese
        'el': ('\u0370', '\u03FF'),  # Greek
        'ku': ('\u0600', '\u06FF'),  # Kurdish (Arabic script)
        'es': ('a-zA-ZáéíóúüñÁÉÍÓÚÜÑ',),  # Spanish
        'fr': ('a-zA-ZàâäéèêëîïôöùûüÿçÀÂÄÉÈÊËÎÏÔÖÙÛÜŸÇ',),  # French
        'en': ('a-zA-Z',)  # English
    }

    def __init__(self, source_lang: str, target_lang: str, 
                 keywords_path: Path = Path("data/keywords/Joshua_Keywords.csv")):
        """Initialize translator with source and target languages."""
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.keywords_path = keywords_path
        self.keyword_dict = self._load_keywords()
        
        # Create language character patterns
        self.source_chars = self._get_language_pattern(source_lang)
        self.target_chars = self._get_language_pattern(target_lang)

    def _get_language_pattern(self, lang: str) -> str:
        """Get regex pattern for language characters."""
        if lang not in self.LANGUAGE_CHARS:
            logger.warning(f"No specific character set defined for {lang}, using default")
            return r'[\w\d_]'
            
        chars = self.LANGUAGE_CHARS[lang]
        if len(chars) == 1:  # Latin-based scripts
            return f'[{chars[0]}]'
        else:  # Unicode ranges
            return f'[{chars[0]}-{chars[1]}]'

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
                    translation_dict[target_word] = source_word  # Add reverse mapping
                    
            # logger.info(f"Loaded bidirectional keywords dictionary for {extract_language(self.source_lang)} "
            #          f"<-> {extract_language(self.target_lang)}")
                     
            return translation_dict
            
        except Exception as e:
            logger.error(f"Error loading keywords: {str(e)}")
            raise

    def translate_code(self, code: str) -> str:
        """Translate code while preserving structure."""
        if not code or not isinstance(code, str):
            return code

        lines = code.split('\n')
        translated_lines = []

        for line in lines:
            # Preserve empty lines with their indentation
            if not line.strip():
                translated_lines.append(line)
                continue

            # Process line
            indentation = self._get_indentation(line)
            stripped_line = line[len(indentation):]
            processed_line = self._process_line(stripped_line)
            translated_lines.append(indentation + processed_line)

        return '\n'.join(translated_lines)

    def _get_indentation(self, line: str) -> str:
        """Extract indentation from line."""
        return line[:len(line) - len(line.lstrip())]

    def _process_line(self, line: str) -> str:
        """Process a single line of code."""
        # Split into code and comment
        code_part, comment = self._split_comment(line)
        
        # Process code and comment separately
        processed_code = self._process_code(code_part)
        processed_comment = self._process_comment(comment)
        
        # Combine processed parts
        if processed_comment:
            return f"{processed_code} {processed_comment}"
        return processed_code

    def _split_comment(self, line: str) -> Tuple[str, str]:
        """Split line into code and comment."""
        code_part, comment = line, ""
        in_string = False
        string_char = None
        
        for i, char in enumerate(line):
            if char in '"\'':
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
            elif char == '#' and not in_string:
                code_part = line[:i].rstrip()
                comment = line[i:]
                break
                
        return code_part, comment

    def _process_code(self, code: str) -> str:
        """Process code part with language-aware tokenization."""
        tokens = self._tokenize(code)
        return self._process_tokens(tokens)

    def _process_comment(self, comment: str) -> str:
        """Process comment part."""
        if not comment:
            return ""
        
        # Keep the # symbol
        if comment.startswith('#'):
            comment_text = comment[1:].strip()
            if comment_text:
                return f"#{self._translate_text(comment_text)}"
        return comment

    def _tokenize(self, code: str) -> List[Dict[str, str]]:
        """Tokenize code with language-aware pattern matching."""
        tokens = []
        i = 0
        
        while i < len(code):
            char = code[i]
            
            # Handle whitespace
            if char.isspace():
                space = self._consume_whitespace(code, i)
                tokens.append({'type': 'space', 'value': space})
                i += len(space)
                continue
                
            # Handle strings
            if char in '"\'':
                string, length = self._consume_string(code, i)
                tokens.append({'type': 'string', 'value': string})
                i += length
                continue
                
            # Handle operators and punctuation
            if char in '()+-*/=<>!,[]{}.:':
                operator, length = self._consume_operator(code, i)
                tokens.append({'type': 'operator', 'value': operator})
                i += length
                continue
                
            # Handle words (including language-specific characters and digits)
            word_pattern = f"{self.source_chars}|{self.target_chars}|\\d"
            if re.match(word_pattern, char) or char == '_':
                word, length = self._consume_word(code, i, word_pattern)
                tokens.append({'type': 'word', 'value': word})
                i += length
                continue
                
            # Skip unknown characters
            i += 1
            
        return tokens

    def _consume_whitespace(self, code: str, start: int) -> str:
        """Consume whitespace characters."""
        space = ''
        i = start
        while i < len(code) and code[i].isspace():
            space += code[i]
            i += 1
        return space

    def _consume_string(self, code: str, start: int) -> Tuple[str, int]:
        """Consume string literal."""
        quote = code[start]
        string = quote
        i = start + 1
        while i < len(code):
            if code[i] == '\\' and i + 1 < len(code):
                string += code[i:i+2]
                i += 2
                continue
            string += code[i]
            if code[i] == quote:
                i += 1
                break
            i += 1
        return string, i - start

    def _consume_operator(self, code: str, start: int) -> Tuple[str, int]:
        """Consume operator or punctuation."""
        char = code[start]
        if start + 1 < len(code):
            next_char = code[start + 1]
            if char + next_char in {'==', '!=', '<=', '>=', '//', '**'}:
                return char + next_char, 2
        return char, 1

    def _consume_word(self, code: str, start: int, word_pattern: str) -> Tuple[str, int]:
        """Consume word with language-specific characters and digits."""
        word = ''
        i = start
        while i < len(code):
            if re.match(word_pattern, code[i]) or code[i] == '_':
                word += code[i]
                i += 1
            else:
                break
        return word, i - start

    def _process_tokens(self, tokens: List[Dict[str, str]]) -> str:
        """Process tokens while preserving structure."""
        result = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if token['type'] == 'space':
                result.append(token['value'])
            elif token['type'] == 'string':
                result.append(token['value'])
            elif token['type'] == 'operator':
                result.append(token['value'])
            elif token['type'] == 'word':
                # Check if it's a method call
                if (i > 0 and tokens[i-1]['value'] == '.' and 
                    i + 1 < len(tokens) and tokens[i+1]['value'] == '('):
                    # Translate method name
                    translated = self._translate_token(token['value'])
                    result.append(translated)
                    i += 1
                    continue
                    
                # Check for function calls
                if (i + 1 < len(tokens) and 
                    tokens[i+1]['type'] == 'operator' and 
                    tokens[i+1]['value'] == '('):
                    # Translate function name
                    translated = self._translate_token(token['value'])
                    result.append(translated)
                    result.append('(')
                    i += 2
                    
                    # Process parameters
                    params = []
                    paren_count = 1
                    while i < len(tokens) and paren_count > 0:
                        if tokens[i]['value'] == '(':
                            paren_count += 1
                        elif tokens[i]['value'] == ')':
                            paren_count -= 1
                        
                        if tokens[i]['type'] == 'word':
                            params.append(self._translate_token(tokens[i]['value']))
                        else:
                            params.append(tokens[i]['value'])
                        i += 1
                    
                    result.append(''.join(params))
                    continue
                    
                # Regular word
                translated = self._translate_token(token['value'])
                result.append(translated)
            
            i += 1
        
        return ''.join(result)

    def _translate_token(self, token: str) -> str:
        """Translate a single token."""
        if not token or token.isspace():
            return token
            
        # Check keyword dictionary
        if token in self.keyword_dict:
            return self.keyword_dict[token]
            
        # Return original token if no translation found
        return token

    def _translate_text(self, text: str) -> str:
        """Translate regular text (like comments)."""
        return text  # Placeholder for potential text translation


def             partial_translate_examples(data_path: Path, source_lang: str, target_lang: str, 
                            start_index: int = 0, stage1_samples: int = 0, stage2_samples: int = 0, 
                            back_translation: bool = False, stage2_model: str = "") -> pd.DataFrame:
    """Translate multiple examples from dataset."""
    try:
        df = pd.read_csv(data_path)
        # logger.info(f"Translating from {source_lang} to {target_lang}")
        
        translator = PartialTranslator(source_lang, target_lang)
        
        total_samples = stage1_samples + stage2_samples
        if start_index + total_samples > len(df):
            logger.warning("Requested range exceeds dataset size. Adjusting start index...")
            start_index = 0
            
        selected_df = df.iloc[start_index:start_index + total_samples]
        translations = []
        
        for _, row in tqdm(pd.DataFrame(selected_df).iterrows(), 
                          total=len(selected_df), 
                          desc="Translating code"):
                          
            input_code = (row[f'{stage2_model}_translated_code'] 
                         if back_translation 
                         else row['English_code'])
                         
            translated_code = translator.translate_code(input_code)
            
            if back_translation:
                translations.append({
                    'English_code': row['English_code'],
                    f'{stage2_model}_translated_code': row[f'{stage2_model}_translated_code'],
                    f'{stage2_model}_partial_back_translated_code': translated_code
                })
            else:
                translations.append({
                    'English_code': row['English_code'],
                    'Partial_translated_code': translated_code
                })
        
        return pd.DataFrame(translations)
        
    except Exception as e:
        logger.error(f"Error in partial translation: {str(e)}")
        raise