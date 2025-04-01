"""
High-level interface for using language-masked models.
"""

import torch
from typing import List, Dict, Optional, Any, Union
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)
import logging

from .masker import MultilingualTokenMasker
from .language_codes import SUPPORTED_LANGUAGES

# Logger setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MultilingualLanguageModel:
    """
    A user-friendly wrapper around a Hugging Face model and the MultilingualTokenMasker.
    Handles model/tokenizer loading and provides methods for generation and configuration.
    """

    def __init__(
        self,
        model_name: str,
        mask_strength: float = 0.8,
        allowed_languages: List[str] = ["JA"],
        token_threshold: float = 0.5,
        cache_tokens: bool = True,
        device: Optional[str] = None,
        model_kwargs: Optional[
            Dict[str, Any]
        ] = None,  # For AutoModelForCausalLM.from_pretrained
        tokenizer_kwargs: Optional[
            Dict[str, Any]
        ] = None,  # For AutoTokenizer.from_pretrained
    ):
        """
        Initializes the MultilingualLanguageModel.

        Args:
            model_name: The name or path of the Hugging Face model to load.
            mask_strength: Initial masking strength (0.0 to 1.0).
            allowed_languages: List of initially allowed language codes.
            token_threshold: Threshold for classifying tokens by language.
            cache_tokens: Whether to cache language-specific token IDs.
            device: The device to load the model onto ('cuda', 'cpu', or None for auto-detect).
            model_kwargs: Additional keyword arguments passed to AutoModelForCausalLM.from_pretrained.
            tokenizer_kwargs: Additional keyword arguments passed to AutoTokenizer.from_pretrained.
        """
        self._model_name = model_name
        self._device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        _model_kwargs = model_kwargs or {}
        _tokenizer_kwargs = tokenizer_kwargs or {}

        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name, **_tokenizer_kwargs
        )

        logger.info(f"Loading model: {model_name} onto device: {self._device}")
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name, **_model_kwargs
        ).to(self._device)
        self.model.eval()  # Set model to evaluation mode

        logger.info("Initializing MultilingualTokenMasker...")
        self.masker = MultilingualTokenMasker(
            tokenizer=self.tokenizer,
            model=self.model,  # Pass model reference for device consistency etc.
            device=self._device,
            default_mask_strength=mask_strength,
            allowed_languages=allowed_languages,
            token_threshold=token_threshold,
            cache_tokens=cache_tokens,
        )
        logger.info("MultilingualLanguageModel initialized successfully.")

    def generate(
        self,
        prompt: str,
        max_length: Optional[
            int
        ] = None,  # Default behavior often depends on model config
        max_new_tokens: Optional[int] = 200,  # More explicit control over output length
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        **kwargs: Any,  # Passthrough for other generate() arguments
    ) -> Union[str, List[str]]:
        """
        Generate text from a prompt using the language-masked model.

        Args:
            prompt: The input text prompt.
            max_length: Max total length (prompt + generated). Overrides max_new_tokens if set.
            max_new_tokens: Max number of new tokens to generate.
            temperature: Sampling temperature. Higher values mean more randomness.
            top_p: Nucleus sampling probability.
            do_sample: Whether to use sampling; set to False for greedy decoding.
            num_return_sequences: Number of sequences to generate.
            **kwargs: Additional arguments passed to the underlying model.generate() method.

        Returns:
            The generated text string, or a list of strings if num_return_sequences > 1.
        """
        logger.debug(f'Generating text for prompt: "{prompt[:50]}..."')

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")  # Use attention mask if available

        # Prepare generate arguments
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": self.tokenizer.eos_token_id,  # Common practice for open-ended generation
            "attention_mask": attention_mask,
            **kwargs,  # Allow overriding defaults or adding more params
        }

        # Override max_new_tokens with max_length if provided
        if max_length is not None:
            generate_kwargs.pop(
                "max_new_tokens", None
            )  # Remove max_new_tokens if max_length is set
            generate_kwargs["max_length"] = max_length

        # Add the logits processor from the masker
        generate_kwargs["logits_processor"] = [self.masker.logits_processor()]

        with torch.no_grad():  # Ensure no gradients are computed during inference
            outputs = self.model.generate(input_ids, **generate_kwargs)

        # Decode the generated sequences
        # Handle batch generation (num_return_sequences > 1)
        decoded_outputs = []
        for i in range(num_return_sequences):
            # Slice output to remove the prompt tokens
            output_sequence = outputs[i, input_ids.shape[-1] :]
            decoded = self.tokenizer.decode(output_sequence, skip_special_tokens=True)
            decoded_outputs.append(decoded.strip())

        logger.debug(f"Generated {len(decoded_outputs)} sequences.")
        return decoded_outputs[0] if num_return_sequences == 1 else decoded_outputs

    # --- Configuration Wrappers ---

    def set_mask_strength(self, strength: float) -> None:
        """Set the language masking strength (0.0 to 1.0)."""
        self.masker.set_mask_strength(strength)

    def set_token_threshold(self, threshold: float) -> None:
        """Set the threshold for classifying tokens by language (0.0 to 1.0)."""
        self.masker.set_token_threshold(threshold)

    def set_languages(self, languages: List[str]) -> None:
        """Set the list of allowed language codes (e.g., ["JA", "EN"])."""
        self.masker.set_allowed_languages(languages)

    def add_language(self, language: str) -> bool:
        """Add a language to the allowed list (e.g., "FR")."""
        return self.masker.add_language(language)

    def remove_language(self, language: str) -> bool:
        """Remove a language from the allowed list (e.g., "ZH")."""
        return self.masker.remove_language(language)

    def get_allowed_languages(self) -> List[str]:
        """Get the currently allowed language codes."""
        return self.masker.allowed_languages

    # --- Accessors and Debug ---

    @property
    def underlying_model(self) -> PreTrainedModel:
        """Access the underlying Hugging Face model."""
        return self.model

    @property
    def underlying_tokenizer(self) -> PreTrainedTokenizer:
        """Access the underlying Hugging Face tokenizer."""
        return self.tokenizer

    @property
    def underlying_masker(self) -> MultilingualTokenMasker:
        """Access the underlying MultilingualTokenMasker instance."""
        return self.masker

    def supported_languages(self) -> Dict[str, str]:
        """Get the dictionary of supported language codes and names."""
        # This information is static and stored in language_codes.py
        return SUPPORTED_LANGUAGES

    def debug_token_classification(
        self, text: str, verbose: bool = True
    ) -> Dict[str, int]:
        """Analyze and print the language classification of tokens in a text."""
        return self.masker.debug_token_classification(text, verbose=verbose)
