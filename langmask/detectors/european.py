"""
Language detection functions for European languages.
"""

import unicodedata


def is_english_char(char: str) -> bool:
    """Check if a character is likely English (Latin alphabet)."""
    # Basic Latin (U+0000 to U+007F) - includes ASCII letters, numbers, punctuation
    # Latin-1 Supplement (U+0080 to U+00FF) - includes common accented letters
    if "\u0000" <= char <= "\u00ff":
        # Exclude control characters but keep letters, numbers, punctuation
        return char.isprintable()
    # Latin Extended-A (U+0100 to U+017F)
    if "\u0100" <= char <= "\u017f":
        return True
    # Latin Extended-B (U+0180 to U+024F)
    if "\u0180" <= char <= "\u024f":
        return True
    # Check via unicodedata name
    name = unicodedata.name(char, "").upper()
    return "LATIN" in name


def is_french_char(char: str) -> bool:
    """Check if a character is common in French (includes English/Latin chars + specifics)."""
    if is_english_char(char):
        return True
    # French specific characters (some might be covered by Latin blocks already)
    fr_specific = "àâçéèêëîïôùûüÿœæÀÂÇÉÈÊËÎÏÔÙÛÜŸŒÆ"
    return char in fr_specific


def is_german_char(char: str) -> bool:
    """Check if a character is common in German (includes English/Latin chars + specifics)."""
    if is_english_char(char):
        return True
    # German specific characters (Umlauts, Eszett)
    de_specific = "äöüßÄÖÜẞ"
    return char in de_specific


def is_spanish_char(char: str) -> bool:
    """Check if a character is common in Spanish (includes English/Latin chars + specifics)."""
    if is_english_char(char):
        return True
    # Spanish specific characters
    es_specific = "áéíóúüñÁÉÍÓÚÜÑ¡¿"
    return char in es_specific


def is_italian_char(char: str) -> bool:
    """Check if a character is common in Italian (includes English/Latin chars + specifics)."""
    if is_english_char(char):
        return True
    # Italian specific accented characters
    it_specific = "àèéìòóùÀÈÉÌÒÓÙ"  # Basic accents
    return char in it_specific


def is_russian_char(char: str) -> bool:
    """Check if a character is Russian (Cyrillic)."""
    # Cyrillic (U+0400 to U+04FF)
    if "\u0400" <= char <= "\u04ff":
        return True
    # Cyrillic Supplement (U+0500 to U+052F)
    if "\u0500" <= char <= "\u052f":
        return True
    # Check via unicodedata name
    name = unicodedata.name(char, "").upper()
    return "CYRILLIC" in name


def is_portuguese_char(char: str) -> bool:
    """Check if a character is common in Portuguese (includes English/Latin chars + specifics)."""
    if is_english_char(char):
        return True
    # Portuguese specific characters
    pt_specific = "áàâãéêíóôõúüçÁÀÂÃÉÊÍÓÔÕÚÜÇ"
    return char in pt_specific


# Add other European language detectors as needed
