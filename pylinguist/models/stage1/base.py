# pylinguist/models/stage1/base.py

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import re
import logging
from ...utils.logger import setup_logger

logger = setup_logger('pylinguist.stage1')

class TranslationError(Exception):
    """Custom exception for translation errors."""
    pass

class CodeComponents:
    """Data class to hold code components."""
    def __init__(self):
        self.strings = []  # Store string literals
        self.comments = []  # Store comments
        self.code_parts = []  # Store code segments
        self.string_map = {}  # Map placeholders to actual strings
        self.comment_map = {}  # Map placeholders to comments
        self.indentation = []  # Store indentation levels

class BaseTranslator(ABC):
    def __init__(self, source_lang: str, target_lang: str, retry_attempts: int = 3):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.retry_attempts = retry_attempts
        self.service_name = "base"

    def handle_compound_word(self, word: str) -> str:
        """
        Handle words containing underscores.
        Example: 'user_input' -> translated('user') + '_' + translated('input')
        """
        if '_' not in word:
            return word

        parts = word.split('_')
        translated_parts = []
        for part in parts:
            if part:  # Skip empty parts
                translated = self.translate_text(part)
                # Remove any spaces in the translated part
                translated = translated.replace(' ', '') if translated else part
                translated_parts.append(translated)

        return '_'.join(translated_parts)

    def extract_components(self, code: str) -> CodeComponents:
        """Extract strings, comments, and code parts."""
        components = CodeComponents()
        
        # Split into lines to preserve structure
        lines = code.split('\n')
        
        for line in lines:
            # Store indentation
            indent = len(line) - len(line.lstrip())
            components.indentation.append(indent)
            line = line.lstrip()
            
            # Extract string literals (prioritize double quotes)
            pattern = r'\"([^\"]*)\"|\'([^\']*)\''
            
            def replace_strings(match):
                string_content = match.group(1) or match.group(2)
                placeholder = f"__STR_{len(components.strings)}__"
                components.strings.append(string_content)
                components.string_map[placeholder] = match.group(0)
                return placeholder

            processed_line = re.sub(pattern, replace_strings, line)
            
            # Handle comments
            if '#' in processed_line:
                code_part, comment = processed_line.split('#', 1)
                comment = comment.strip()
                if comment:
                    placeholder = f"__CMT_{len(components.comments)}__"
                    components.comments.append(comment)
                    components.comment_map[placeholder] = f"# {comment}"
                    processed_line = code_part + " " + placeholder
            
            components.code_parts.append(processed_line)
            
        return components

    @abstractmethod
    def translate_text(self, text: str) -> str:
        """Implement actual translation logic in subclasses."""
        pass

    def translate_code(self, code: str) -> str:
        """
        Translate code while handling strings, comments, and compound words.
        """
        try:
            # Extract components
            components = self.extract_components(code)
            
            # Translate strings (as complete sentences)
            translated_strings = {}
            for i, string in enumerate(components.strings):
                translated = self.translate_text(string)
                placeholder = f"__STR_{i}__"
                if translated:
                    # Preserve original quotes
                    original = components.string_map[placeholder]
                    quote = original[0]  # Get the quote character used
                    translated_strings[placeholder] = f'{quote}{translated}{quote}'
                else:
                    translated_strings[placeholder] = components.string_map[placeholder]

            # Translate comments (as complete sentences)
            translated_comments = {}
            for i, comment in enumerate(components.comments):
                translated = self.translate_text(comment)
                placeholder = f"__CMT_{i}__"
                if translated:
                    translated_comments[placeholder] = f"# {translated}"
                else:
                    translated_comments[placeholder] = components.comment_map[placeholder]

            # Process each line
            translated_lines = []
            for line, indent in zip(components.code_parts, components.indentation):
                # Handle compound words
                words = line.split()
                translated_words = []
                
                for word in words:
                    # Skip if it's a placeholder
                    if word.startswith('__STR_') or word.startswith('__CMT_'):
                        translated_words.append(word)
                    else:
                        # Handle compound words
                        translated = self.handle_compound_word(word)
                        translated_words.append(translated)

                processed_line = ' '.join(translated_words)
                
                # Replace string placeholders
                for placeholder, translation in translated_strings.items():
                    processed_line = processed_line.replace(placeholder, translation)
                    
                # Replace comment placeholders
                for placeholder, translation in translated_comments.items():
                    processed_line = processed_line.replace(placeholder, translation)
                
                # Restore indentation
                translated_lines.append(' ' * indent + processed_line)

            return '\n'.join(translated_lines)

        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return code

    @abstractmethod
    def get_supported_languages(self) -> Dict[str, List[str]]:
        """Get supported language pairs."""
        pass

    @abstractmethod
    def validate_languages(self) -> bool:
        """Validate language pair is supported."""
        pass

    def get_service_info(self) -> Dict:
        """Get information about the translation service."""
        return {
            'service_name': self.service_name,
            'source_lang': self.source_lang,
            'target_lang': self.target_lang,
            'retry_attempts': self.retry_attempts
        }