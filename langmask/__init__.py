"""
LangMask Package: Control LLM output languages.
"""

__version__ = "0.1.0"  # Initial version

from .detectors import (  # ... add other detector functions if they should be public
    is_chinese_char,
    is_english_char,
    is_japanese_char,
    is_korean_char,
    is_language_token,
)
from .language_codes import SUPPORTED_LANGUAGES

# Import key classes/functions for easy access
from .masker import MultilingualTokenMasker
from .model import MultilingualLanguageModel  # Now available

__all__ = [
    "MultilingualTokenMasker",
    "MultilingualLanguageModel",  # Export the new class
    "SUPPORTED_LANGUAGES",
    "is_language_token",
    "is_japanese_char",
    "is_chinese_char",
    "is_korean_char",
    "is_english_char",
    "__version__",
]
