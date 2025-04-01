"""
Utility functions for language detection in tokens.
"""

from typing import Callable, Dict

from .asian import is_chinese_char, is_japanese_char, is_korean_char
from .european import (
    is_english_char,
    is_french_char,
    is_german_char,
    is_italian_char,
    is_portuguese_char,
    is_russian_char,
    is_spanish_char,
)

# Mapping from language code to character detection function
LANG_DETECTORS: Dict[str, Callable[[str], bool]] = {
    "JA": is_japanese_char,
    "ZH": is_chinese_char,
    "KO": is_korean_char,
    "EN": is_english_char,
    "FR": is_french_char,
    "DE": is_german_char,
    "ES": is_spanish_char,
    "IT": is_italian_char,
    "RU": is_russian_char,
    "PT": is_portuguese_char,
    # Add other languages here
}


def get_language_char_detector(lang: str) -> Callable[[str], bool]:
    """Get the character detection function for a given language code."""
    # Default to English if the language is not specifically handled
    return LANG_DETECTORS.get(lang, is_english_char)


def is_language_token(token: str, lang: str, threshold: float = 0.5) -> bool:
    """Check if a token belongs primarily to a given language."""
    if not token:
        return False

    # Handle byte-level tokenizers or special prefixes
    decoded_token = token
    if token.startswith("Ġ"):  # GPT-2/RoBERTa style
        decoded_token = token[1:]
    elif token.startswith(" "):  # SentencePiece/T5 style
        decoded_token = token[1:]
    elif token.startswith("##"):  # BERT style wordpiece
        decoded_token = token[2:]

    # Treat whitespace and pure numbers as language-neutral (allow in any language context)
    if decoded_token.isspace() or decoded_token.isdigit() or token == "\n":
        return True

    # Treat common punctuation as language-neutral
    # More sophisticated handling might be needed for language-specific punctuation
    if len(decoded_token) == 1 and not decoded_token.isalnum():
        return True

    if not decoded_token:  # After removing prefix, token might be empty
        return False

    # Get the appropriate character detector for the language
    is_lang_char = get_language_char_detector(lang)

    # Count characters belonging to the target language
    lang_char_count = sum(1 for char in decoded_token if is_lang_char(char))

    # Check if the proportion of language-specific characters meets the threshold
    # Avoid division by zero for empty strings after decoding
    try:
        proportion = lang_char_count / len(decoded_token)
    except ZeroDivisionError:
        return False  # Or True, depending on desired behavior for empty decoded tokens

    return proportion >= threshold
