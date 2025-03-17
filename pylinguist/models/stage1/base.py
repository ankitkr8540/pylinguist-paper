from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import re
from ...utils.logger import setup_logger

logger = setup_logger('pylinguist.stage1.base')

class BaseTranslator(ABC):
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

    def __init__(self, source_lang: str, target_lang: str):
        self.source_lang = source_lang
        self.target_lang = target_lang
        
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
    
    def is_target_language(self, text: str) -> bool:
        """Check if text is already in target language."""
        # For non-Latin scripts, check if any character is in target language range
        if len(self.LANGUAGE_CHARS.get(self.target_lang, [])) == 2:
            target_start, target_end = self.LANGUAGE_CHARS[self.target_lang]
            for char in text:
                if target_start <= char <= target_end:
                    return True
            return False
        
        # For Latin scripts with special characters, check target language pattern
        target_pattern = self._get_language_pattern(self.target_lang)
        for char in text:
            if re.match(target_pattern, char) and not re.match(r'[a-zA-Z0-9_]', char):
                return True
                
        # Could be further improved with language detection libraries
        return False
    
    def translate_code(self, code: str) -> str:
        """Translate code while preserving structure."""
        if not code or not isinstance(code, str):
            return code

        try:
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
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return code

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
                # Skip translation if already in target language
                if self.is_target_language(comment_text):
                    return comment
                return f"# {self.translate_text(comment_text)}"
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
                word = token['value']
                
                # Skip translation if already in target language
                if self.is_target_language(word):
                    result.append(word)
                    i += 1
                    continue
                    
                # Check if it's a method call
                if (i > 0 and tokens[i-1]['value'] == '.' and 
                    i + 1 < len(tokens) and tokens[i+1]['value'] == '('):
                    # Translate method name
                    translated = self._translate_token(word)
                    result.append(translated)
                elif (i + 1 < len(tokens) and 
                      tokens[i+1]['type'] == 'operator' and 
                      tokens[i+1]['value'] == '('):
                    # Translate function name
                    translated = self._translate_token(word)
                    result.append(translated)
                else:
                    # Regular word - check for snake_case
                    if '_' in word:
                        parts = word.split('_')
                        translated_parts = []
                        for part in parts:
                            if self.is_target_language(part):
                                translated_parts.append(part)  # Skip if already translated
                            else:
                                translated_parts.append(self._translate_token(part))
                        result.append('_'.join(translated_parts))
                    else:
                        translated = self._translate_token(word)
                        result.append(translated)
            
            i += 1
        
        return ''.join(result)

    def _translate_token(self, token: str) -> str:
        """Translate a single token."""
        if not token or token.isspace():
            return token
            
        # Skip translation if already in target language
        if self.is_target_language(token):
            return token
            
        # Skip common programming keywords that shouldn't be translated
        common_keywords = {'if', 'else', 'for', 'while', 'def', 'class', 'return', 'import', 
                         'from', 'as', 'try', 'except', 'finally', 'with', 'in', 'is', 'not',
                         'and', 'or', 'True', 'False', 'None', 'print', 'input', 'len', 'super'}
        if token in common_keywords:
            return token
            
        # Implement your token translation logic here
        # This might involve using your translate_text method
        return self.translate_text(token)

    @abstractmethod
    def translate_text(self, text: str) -> str:
        """Translate text from source to target language."""
        pass