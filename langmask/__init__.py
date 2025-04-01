"""
LangMask Package: Control LLM output languages.
"""

__version__ = "0.1.0"  # Initial version

# Import key classes/functions for easy access
from .masker import MultilingualTokenMasker
from .model import MultilingualLanguageModel  # Now available
from .language_codes import SUPPORTED_LANGUAGES
from .detectors import (
    is_language_token,
    is_japanese_char,
    is_chinese_char,
    is_korean_char,
    is_english_char,
    # ... add other detector functions if they should be public
)

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
