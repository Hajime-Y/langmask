"""
Core token masking logic for LangMask.
"""

import logging
from typing import Callable, Dict, List, Optional, Set

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .detectors import is_language_token
from .language_codes import SUPPORTED_LANGUAGES

# Logger setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MultilingualTokenMasker:
    """
    Manages token masking based on allowed languages.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        default_mask_strength: float = 0.8,
        allowed_languages: List[str] = ["JA"],  # Default to Japanese
        token_threshold: float = 0.5,  # Threshold for language detection in a token
        cache_tokens: bool = True,
    ):
        """
        Initializes the MultilingualTokenMasker.

        Args:
            tokenizer: The Hugging Face tokenizer.
            model: The Hugging Face model (optional for direct generation here, but used for device).
            device: Device to run computations on ("cuda" or "cpu").
            default_mask_strength: Default masking strength (0.0 to 1.0).
            allowed_languages: List of initially allowed language codes (e.g., ["JA", "EN"]).
            token_threshold: Minimum proportion of language-specific chars for a token to be classified as that language.
            cache_tokens: Whether to cache language-specific token IDs.
        """
        self.tokenizer = tokenizer
        self.model = model  # Keep model reference for device consistency if needed
        self.device = device
        self.mask_strength = max(0.0, min(1.0, default_mask_strength))
        self.token_threshold = token_threshold

        # Validate and set allowed languages
        self.allowed_languages = []
        for lang in allowed_languages:
            if lang in SUPPORTED_LANGUAGES:
                self.allowed_languages.append(lang)
            else:
                logger.warning(f"Language code '{lang}' is not supported. Ignoring.")

        if not self.allowed_languages:
            logger.warning("No valid languages specified. Defaulting to Japanese (JA).")
            self.allowed_languages = ["JA"]

        logger.info(
            f"Allowed languages: {', '.join([f'{lang} ({SUPPORTED_LANGUAGES[lang]})' for lang in self.allowed_languages])}"
        )

        # Token cache and special tokens
        self._token_cache: Dict[str, Set[int]] = {}
        self._special_token_ids: Set[int] = self._get_special_token_ids()

        # Pre-cache tokens for allowed languages if requested
        if cache_tokens:
            for lang in self.allowed_languages:
                if lang not in self._token_cache:
                    logger.info(f"Caching tokens for language: {lang}")
                    self._token_cache[lang] = self._identify_language_tokens(lang)

    def _get_special_token_ids(self) -> Set[int]:
        """Get IDs of special tokens from the tokenizer."""
        special_token_ids = set()

        # Common special tokens
        special_token_attrs = [
            "bos_token",
            "eos_token",
            "unk_token",
            "sep_token",
            "pad_token",
            "cls_token",
            "mask_token",
        ]
        for attr in special_token_attrs:
            if hasattr(self.tokenizer, attr):
                token = getattr(self.tokenizer, attr)
                if token:
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                    # Ensure the token ID is valid and not the UNK token ID unless it's the only representation
                    if (
                        token_id is not None
                        and token_id != self.tokenizer.unk_token_id
                        or token == self.tokenizer.unk_token
                    ):
                        special_token_ids.add(token_id)

        # Additional special tokens
        if hasattr(self.tokenizer, "additional_special_tokens"):
            additional_tokens = getattr(self.tokenizer, "additional_special_tokens")
            if isinstance(additional_tokens, list):
                for token in additional_tokens:
                    if token:
                        token_id = self.tokenizer.convert_tokens_to_ids(token)
                        if (
                            token_id is not None
                            and token_id != self.tokenizer.unk_token_id
                            or token == self.tokenizer.unk_token
                        ):
                            special_token_ids.add(token_id)

        # Check the vocab directly for tokens like <|endoftext|> if not caught above
        # This is less reliable but can be a fallback
        # Example: Check for common control tokens if not captured by attributes
        # common_control_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "[EOS]"]
        # for token in common_control_tokens:
        #     token_id = self.tokenizer.convert_tokens_to_ids(token)
        #     if token_id is not None and token_id != self.tokenizer.unk_token_id:
        #          special_token_ids.add(token_id)

        # Ensure None is not added if convert_tokens_to_ids returns None
        special_token_ids.discard(None)

        logger.debug(f"Identified special token IDs: {special_token_ids}")
        return special_token_ids

    def _identify_language_tokens(self, lang: str) -> Set[int]:
        """Identify all token IDs associated with a specific language."""
        if lang not in SUPPORTED_LANGUAGES:
            logger.warning(f"Cannot identify tokens for unsupported language: {lang}")
            return set()

        language_token_ids = set()
        # Include special tokens by default as they are language-agnostic controls
        language_token_ids.update(self._special_token_ids)

        vocab = self.tokenizer.get_vocab()
        for token_str, token_id in vocab.items():
            # Skip if already identified as a special token
            if token_id in self._special_token_ids:
                continue

            # Use the utility function from detectors
            if is_language_token(token_str, lang, threshold=self.token_threshold):
                language_token_ids.add(token_id)

        logger.debug(f"Identified {len(language_token_ids)} tokens for language {lang}")
        return language_token_ids

    def _get_allowed_token_ids(self) -> Set[int]:
        """Get the combined set of token IDs for all currently allowed languages."""
        allowed_token_ids = set()
        # Always include special tokens
        allowed_token_ids.update(self._special_token_ids)

        for lang in self.allowed_languages:
            if lang in self._token_cache:
                allowed_token_ids.update(self._token_cache[lang])
            else:
                # Identify and cache if not already done
                logger.info(f"Caching tokens for language: {lang}")
                lang_tokens = self._identify_language_tokens(lang)
                self._token_cache[lang] = lang_tokens
                allowed_token_ids.update(lang_tokens)

        return allowed_token_ids

    def apply_language_mask(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply the language mask to the logits tensor.

        Args:
            logits: The raw logits tensor from the model (batch_size, vocab_size).

        Returns:
            The modified logits tensor with disallowed tokens penalized.
        """
        if self.mask_strength <= 0:
            return logits  # No masking

        allowed_token_ids = self._get_allowed_token_ids()

        # Create mask: 0 for allowed tokens, 1 for disallowed tokens
        vocab_size = logits.shape[-1]
        # Ensure all allowed IDs are within the vocab size
        valid_allowed_ids = [idx for idx in allowed_token_ids if idx < vocab_size]
        if len(valid_allowed_ids) != len(allowed_token_ids):
            logger.warning(
                "Some allowed token IDs are outside the tokenizer's vocabulary size."
            )

        disallowed_mask = torch.ones(
            vocab_size, device=self.device, dtype=torch.float32
        )
        if valid_allowed_ids:
            # Use torch.tensor to initialize with valid_allowed_ids, ensuring correct device and type
            allowed_indices = torch.tensor(
                valid_allowed_ids, device=self.device, dtype=torch.long
            )
            disallowed_mask[allowed_indices] = 0.0

        # Calculate penalty
        penalty: torch.Tensor  # Add type annotation
        if self.mask_strength >= 0.9999:  # Use a threshold for hard masking
            penalty = torch.tensor(
                -torch.finfo(logits.dtype).max, device=self.device
            )  # Ensure tensor is on correct device
        else:
            # Scale penalty based on mask strength.
            # Using torch.log for calculation, ensure the input is a tensor
            penalty_factor = 100.0  # Adjust this factor based on testing
            penalty = penalty_factor * torch.log(
                torch.tensor(1.0 - self.mask_strength + 1e-9, device=self.device)
            )
        penalty = penalty.to(logits.dtype)  # Ensure penalty matches logits dtype

        # Apply penalty: Add a large negative value to disallowed token logits
        # Ensure broadcasting works correctly (disallowed_mask needs to match the last dim of logits)
        modified_logits = logits + (disallowed_mask * penalty)

        return modified_logits

    def logits_processor(
        self,
    ) -> Callable[
        [torch.Tensor, torch.Tensor], torch.Tensor
    ]:  # Add return type annotation
        """Return a callable logits processor compatible with Hugging Face's generate."""

        def process_logits(
            input_ids: torch.Tensor, logits: torch.Tensor
        ) -> torch.Tensor:  # Add argument type annotations and return type
            # Logits shape: (batch_size, sequence_length, vocab_size)
            # We need to apply the mask to the last prediction logits
            # In HF generate, logits passed here are usually (batch_size, vocab_size)
            # for the next token prediction.
            if logits.ndim == 2:  # Shape (batch_size, vocab_size)
                return self.apply_language_mask(logits)
            elif logits.ndim == 3:  # Shape (batch_size, sequence_length, vocab_size)
                # Apply mask only to the last token's logits in the sequence
                last_logits = logits[:, -1, :]
                masked_last_logits = self.apply_language_mask(last_logits)
                logits[:, -1, :] = masked_last_logits
                return logits
            else:
                logger.error(f"Unexpected logits shape: {logits.shape}")
                return logits  # Return unmodified if shape is unexpected

        return process_logits

    # --- Configuration Methods ---

    def set_mask_strength(self, strength: float) -> None:
        """Set the masking strength (0.0 to 1.0)."""
        self.mask_strength = max(0.0, min(1.0, strength))
        logger.info(f"Mask strength set to: {self.mask_strength:.2f}")

    def set_token_threshold(self, threshold: float) -> None:
        """Set the threshold for classifying tokens (0.0 to 1.0)."""
        self.token_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Token language threshold set to: {self.token_threshold:.2f}")
        # Invalidate cache as thresholds changed definition of language tokens
        self._token_cache = {}
        logger.info(
            "Token cache cleared due to threshold change. Will re-cache on next use."
        )
        # Re-cache immediately for currently allowed languages if desired
        # for lang in self.allowed_languages:
        #    self._token_cache[lang] = self._identify_language_tokens(lang)

    def set_allowed_languages(self, languages: List[str]) -> None:
        """Set the list of allowed language codes."""
        valid_languages = []
        for lang in languages:
            if lang in SUPPORTED_LANGUAGES:
                valid_languages.append(lang)
            else:
                logger.warning(f"Language code '{lang}' is not supported. Ignoring.")

        if not valid_languages:
            logger.warning("No valid languages provided. Keeping current settings.")
            return

        self.allowed_languages = valid_languages
        logger.info(
            f"Allowed languages set to: {', '.join([f'{lang} ({SUPPORTED_LANGUAGES[lang]})' for lang in self.allowed_languages])}"
        )

        # Ensure newly allowed languages are cached if needed
        for lang in self.allowed_languages:
            if lang not in self._token_cache:
                logger.info(f"Caching tokens for newly allowed language: {lang}")
                self._token_cache[lang] = self._identify_language_tokens(lang)

    def add_language(self, language: str) -> bool:
        """Add a language to the list of allowed languages."""
        if language not in SUPPORTED_LANGUAGES:
            logger.warning(f"Language code '{language}' is not supported. Cannot add.")
            return False

        if language in self.allowed_languages:
            logger.info(f"Language {language} is already allowed.")
            return True

        self.allowed_languages.append(language)
        logger.info(
            f"Language {language} ({SUPPORTED_LANGUAGES[language]}) added to allowed list."
        )

        # Cache tokens for the newly added language
        if language not in self._token_cache:
            logger.info(f"Caching tokens for newly added language: {language}")
            self._token_cache[language] = self._identify_language_tokens(language)
        return True

    def remove_language(self, language: str) -> bool:
        """Remove a language from the list of allowed languages."""
        if language not in self.allowed_languages:
            logger.info(f"Language {language} is not in the allowed list.")
            return False

        if len(self.allowed_languages) <= 1:
            logger.warning(
                "Cannot remove the last allowed language. At least one language must be allowed."
            )
            return False

        self.allowed_languages.remove(language)
        logger.info(
            f"Language {language} ({SUPPORTED_LANGUAGES[language]}) removed from allowed list."
        )
        # Note: We don't remove from cache, as it might be added back later.
        return True

    # --- Debugging Methods ---

    def debug_token_classification(
        self, text: str, verbose: bool = True
    ) -> Dict[str, int]:
        """
        Analyze tokens in a text and classify them by language.

        Args:
            text: The text to analyze.
            verbose: Whether to print detailed classification results.

        Returns:
            A dictionary mapping language codes to the count of tokens classified for that language.
        """
        if not text:
            return {}

        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        classification: Dict[str, int] = {
            lang: 0 for lang in SUPPORTED_LANGUAGES
        }  # Add type annotation
        classification["UNK"] = 0  # For unknown or mixed tokens
        classification["SPECIAL"] = 0  # For special tokens

        special_ids = self._special_token_ids

        for token_str, token_id in zip(tokens, token_ids):
            if token_id in special_ids:
                classification["SPECIAL"] += 1
                continue

            # Basic check for neutrals first (can customize is_language_token for this)
            decoded_token = token_str
            if token_str.startswith("Ġ"):
                decoded_token = token_str[1:]
            elif token_str.startswith(" "):
                decoded_token = token_str[1:]
            elif token_str.startswith("##"):
                decoded_token = token_str[2:]

            if not decoded_token:  # Handle cases like 'Ġ' itself if it's a token
                classification["SPECIAL"] += 1  # Treat as special/control
                continue

            if (
                decoded_token.isspace()
                or decoded_token.isdigit()
                or (len(decoded_token) == 1 and not decoded_token.isalnum())
            ):
                classification["NEUTRAL"] += 1
                continue

            classified = False
            for lang_code in SUPPORTED_LANGUAGES:
                if is_language_token(
                    token_str, lang_code, threshold=self.token_threshold
                ):
                    classification[lang_code] += 1
                    classified = True
                    # Decide if a token can belong to multiple languages or just the first match
                    # break # Uncomment for first match only

            if not classified:
                classification["UNK"] += 1

        # Calculate stats
        stats = {lang: count for lang, count in classification.items()}
        total_tokens = len(tokens)

        if verbose:
            print(f"--- Token Classification Debug ---")
            print(f'Text: "{text}"')
            print(f"Total Tokens: {total_tokens}")
            for lang_code, count in stats.items():
                if count > 0:
                    percentage = (count / total_tokens * 100) if total_tokens > 0 else 0
                    lang_name = SUPPORTED_LANGUAGES.get(
                        lang_code, lang_code
                    )  # Use code if not in map (e.g., SPECIAL)
                    print(
                        f"  {lang_code} ({lang_name}): {count} tokens ({percentage:.1f}%)"
                    )
            print(f"------------------------------------")

        return stats
