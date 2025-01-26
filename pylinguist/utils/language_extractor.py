import logging
from pylinguist.utils.logger import setup_logger

logger = setup_logger()

JOSHUA_KEYWORD = {
        'fr': 'FrenchKey',
        'es': 'SpanishKey',
        'ku': 'KurdishKey',
        'en': 'EnglishKey',
        'hi': 'HindiKey',
        'bn': 'BengaliKey',
        'zh': 'MandarinKey',
        'el': 'GreekKey'
    }

LANGUAGE_CODES = {
    'fr': 'French',
    'es': 'Spanish',
    'ku': 'Kurdish',
    'en': 'English',
    'hi': 'Hindi',
    'bn': 'Bengali',
    'zh': 'Mandarin',
    'el': 'Greek'
}


# extract language based on the google language codes
def extract_keyword_header(code):
    """Extract language from Google language code."""
    return JOSHUA_KEYWORD.get(code, f'{code} Key not found')

def extract_language(code):
    """Extract language from Google language code."""
    return LANGUAGE_CODES.get(code, f'{code} Key not found')
    