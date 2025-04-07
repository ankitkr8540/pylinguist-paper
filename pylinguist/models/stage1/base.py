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
            return r'[\w\d_]' #
            
        chars = self.LANGUAGE_CHARS[lang]
        if len(chars) == 1:  # Latin-based scripts
            return f'[{chars[0]}]'
        else:  # Unicode ranges
            return f'[{chars[0]}-{chars[1]}]'
    
    def is_target_language(self, text: str) -> bool:
        """Check if text is already in target language."""
        # Check if text is a number
        if text.isdigit():
            return True
        
        # For English as target language
        if self.target_lang == 'en':
            # Check if word is mostly English letters
            english_chars = 0
            for char in text:
                if char.isalpha() and ord('a') <= ord(char.lower()) <= ord('z'):
                    english_chars += 1
                    return True
            return False
        
        # For Hindi and other non-Latin scripts
        elif self.target_lang in ['hi', 'bn', 'zh', 'el', 'ku']:
            # Get Unicode range for target language
            if len(self.LANGUAGE_CHARS.get(self.target_lang, [])) == 2:
                target_start, target_end = self.LANGUAGE_CHARS[self.target_lang]
                
                # Check each character to see if it's in the target language Unicode range
                for char in text:
                    # Skip spaces, underscores, and digits
                    if char.isspace() or char == '_' or char.isdigit():
                        continue
                        
                    # If any character is in the target language range, the word is in target language
                    if target_start <= char <= target_end:
                        return True
        
        # For Latin-based languages like Spanish, French
        elif self.target_lang in ['es', 'fr']:
            target_pattern = self._get_language_pattern(self.target_lang)
            for char in text:
                if re.match(target_pattern, char) and not re.match(r'[a-zA-Z0-9_]', char):
                    return True
        
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
            if code[i] in '()[]{}+-*/%=<>!&|^~.,;:?@#$\'"\\':
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
    
        # Try to match three-character operators first
        if start + 3 <= len(code):
            three_chars = code[start:start+3]
            if three_chars in {'<<=', '>>=', '...'}:
                return three_chars, 3

        # Then try two-character operators
        if start + 2 <= len(code):
            two_chars = code[start:start+2]
            if two_chars in {
                '==', '!=', '<=', '>=', '//', '**',
                '+=', '-=', '*=', '/=', '%=', '&=',
                '|=', '^=', '<<', '>>', '&&', '||',
                '++', '--', '->', '=>', '::'
            }:
                return two_chars, 2

        # Otherwise, it's a single-character operator
        return code[start], 1

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
        result = []
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            if token['type'] in ('space', 'operator', 'string'):
                # Preserve spaces, operators, and strings as is
                result.append(token['value'])
            elif token['type'] == 'word':
                word = token['value']
                
                # Skip translation if already in target language
                if self.is_target_language(word):
                    result.append(word)
                    i += 1
                    continue
                
                # Skip translation for special identifiers
                skip_words = {'__init__', '__name__', '__main__', 'self', 'cls', 'args', 'kwargs'}
                if word in skip_words or word.isdigit():
                    result.append(word)
                    i += 1
                    continue
                
                # Process identifiers - variables, functions, etc.
                translated = None
                
                # Check if this is a function/method call
                is_function = (i + 1 < len(tokens) and 
                            tokens[i+1]['type'] == 'operator' and 
                            tokens[i+1]['value'] == '(')
                
                # Handle translation based on word structure
                if '_' in word:
                    # For compound words, translate the entire phrase
                    phrase = word.replace('_', ' ')
                    translated = self.translate_text(phrase)
                else:
                    # For simple words
                    translated = self.translate_text(word)
                
                # Critical fix: Always replace spaces with underscores for identifiers
                if translated:
                    # Force underscore format for identifiers to maintain valid Python syntax
                    formatted = translated.replace(' ', '_')
                    result.append(formatted)
                else:
                    result.append(word)
                    
            i += 1
        
        return ''.join(result)
    def _translate_token(self, token: str) -> str:
        """Translate a single token with improved handling for compound words."""
        if not token or token.isspace():
            return token
            
        # Skip translation if already in target language
        if self.is_target_language(token):
            return token
        
        # Handle compound words with underscores (snake_case)
        if '_' in token:
            # Convert the entire token to a space-separated phrase for better translation
            phrase = token.replace('_', ' ')
            
            # Translate the entire phrase as a single semantic unit
            translated_phrase = self.translate_text(phrase)
            
            # If translation succeeded, convert spaces back to underscores
            if translated_phrase:
                return translated_phrase.replace(' ', '_')
            return token
        
        # For simple tokens, translate directly
        translated = self.translate_text(token)
        
        # Ensure no spaces in the result (convert to underscores if needed)
        if translated and ' ' in translated:
            return translated.replace(' ', '_')
        return translated or token

    @abstractmethod
    def translate_text(self, text: str) -> str:
        """Translate text from source to target language."""
        pass