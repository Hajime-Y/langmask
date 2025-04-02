"""
High-level interface for using language-masked models.
"""

import logging
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union, cast

import torch
from torch import device as torch_device
from transformers import (
    AutoModelForCausalLM,
    GenerationMixin,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.generation.utils import GenerateOutput
from transformers.modeling_utils import GenerationMixin as BaseGenerationMixin
from transformers.modeling_utils import PreTrainedModel as BasePreTrainedModel

from .language_codes import SUPPORTED_LANGUAGES
from .masker import MultilingualTokenMasker

# Logger setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

T = TypeVar("T", bound="PreTrainedModel")


class MultilingualLanguageModel(PreTrainedModel, Generic[T]):
    """
    A language-masked wrapper around a Hugging Face model.
    Inherits from PreTrainedModel to maintain compatibility with the Hugging Face ecosystem.
    """

    def __init__(
        self,
        model: T,
        tokenizer: PreTrainedTokenizer,
        mask_strength: float = 0.8,
        allowed_languages: List[str] = ["JA"],
        token_threshold: float = 0.5,
        cache_tokens: bool = True,
    ):
        """
        Initializes the MultilingualLanguageModel.

        Args:
            model: The base Hugging Face model to wrap.
            tokenizer: The tokenizer associated with the model.
            mask_strength: Initial masking strength (0.0 to 1.0).
            allowed_languages: List of initially allowed language codes.
            token_threshold: Threshold for classifying tokens by language.
            cache_tokens: Whether to cache language-specific token IDs.
        """
        # Initialize parent class with the base model's config
        super().__init__(model.config)

        self.base_model: T = model
        self._device: torch_device = model.device
        self.config = model.config

        logger.info("Initializing MultilingualTokenMasker...")
        self.masker = MultilingualTokenMasker(
            tokenizer=tokenizer,
            model=model,
            device=str(self._device),  # Convert device to string
            default_mask_strength=mask_strength,
            allowed_languages=allowed_languages,
            token_threshold=token_threshold,
            cache_tokens=cache_tokens,
        )
        logger.info("MultilingualLanguageModel initialized successfully.")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Forward pass of the model.
        Delegates to the base model's forward pass.
        """
        output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        return cast(Dict[str, Any], output)

    def prepare_inputs_for_generation(
        self, input_ids: torch.Tensor, **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Prepare inputs for generation.
        Delegates to the base model's preparation method.
        """
        output = self.base_model.prepare_inputs_for_generation(input_ids, **kwargs)
        return cast(Dict[str, Any], output)

    def _reorder_cache(self, past: Any, beam_idx: torch.Tensor) -> Any:
        """
        Reorder the cache for beam search.
        Delegates to the base model's cache reordering method.
        """
        return self.base_model._reorder_cache(past, beam_idx)

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Generate text using the language-masked model.
        Accepts the same arguments as the base model's generate method.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            **kwargs: Additional arguments passed to the base model's generate method.

        Returns:
            Generated token IDs or a dictionary containing generation information.
        """
        # Add the logits processor from the masker
        kwargs["logits_processor"] = kwargs.get("logits_processor", [])
        kwargs["logits_processor"].append(self.masker.logits_processor())

        # Ensure pad_token_id and eos_token_id are set if not provided
        if "pad_token_id" not in kwargs and hasattr(self.config, "pad_token_id"):
            kwargs["pad_token_id"] = self.config.pad_token_id
        if "eos_token_id" not in kwargs and hasattr(self.config, "eos_token_id"):
            kwargs["eos_token_id"] = self.config.eos_token_id

        output = self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        return cast(Union[torch.Tensor, Dict[str, Any]], output)

    # --- Configuration Methods ---

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

    # --- Debug Methods ---

    def debug_token_classification(
        self, text: str, tokenizer: PreTrainedTokenizer, verbose: bool = True
    ) -> Dict[str, int]:
        """Analyze and print the language classification of tokens in a text."""
        return self.masker.debug_token_classification(text, verbose=verbose)

    @property
    def device(self) -> torch_device:
        """Get the device the model is on."""
        return self._device

    @device.setter
    def device(self, device: Union[str, torch_device]) -> None:
        """Set the device for the model."""
        self._device = torch.device(device)
        self.base_model.to(self._device)
