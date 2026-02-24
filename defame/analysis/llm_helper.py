"""
Generic LLM helper for analyzers.

This module provides a lightweight wrapper around DEFAME's Model class
for use in analyzers. It handles the infrastructure of calling LLMs,
while the analyzer contains the domain-specific prompts and logic.
"""

from typing import Any
from defame.common.modeling import Model, make_model
from defame.common.prompt import Prompt
import re


class AnalyzerLLMHelper:
    """
    Generic helper class for making LLM calls in analyzers.

    This provides a thin wrapper around DEFAME's Model class with
    convenience methods for common patterns (rating, classification, etc.).
    Domain-specific prompts should live in the analyzer, not here.

    Usage:
        # In analyzer __init__:
        helper = AnalyzerLLMHelper.from_config(config)

        # In analyzer methods:
        prompt_text = "Your analyzer-specific prompt here..."
        rating = helper.get_numeric_rating(prompt_text, scale=(1, 5))
        classification = helper.classify(prompt_text, options=["yes", "no"])
    """

    def __init__(self, model: Model):
        """
        Initialize the helper with a DEFAME Model instance.

        Args:
            model: A DEFAME Model instance (e.g., from make_model())
        """
        self.model = model

    @classmethod
    def from_model(cls, model: Model) -> "AnalyzerLLMHelper":
        """
        Create an AnalyzerLLMHelper from a Model instance.

        This is the preferred way when you have an already-instantiated model.

        Args:
            model: DEFAME Model instance

        Returns:
            AnalyzerLLMHelper instance
        """
        return cls(model)

    @classmethod
    def from_config_dict(cls, config: dict[str, Any]) -> "AnalyzerLLMHelper":
        """
        Create an AnalyzerLLMHelper from a config dictionary.

        This is a legacy method for backwards compatibility.
        Prefer using from_model() or passing AnalyzerConfig instead.

        Expected config format:
        {
            "llm_model": "gpt4o_mini",  # or any model shorthand/specifier
            "llm_temperature": 0.3,  # optional, defaults to model's default
            "llm_max_response_len": 512,  # optional
        }

        Args:
            config: Configuration dictionary

        Returns:
            AnalyzerLLMHelper instance
        """
        model_name = config.get("llm_model", "gpt4o_mini")

        # Extract model kwargs from config
        model_kwargs = {}
        if "llm_temperature" in config:
            model_kwargs["temperature"] = config["llm_temperature"]
        if "llm_max_response_len" in config:
            model_kwargs["max_response_len"] = config["llm_max_response_len"]

        model = make_model(model_name, **model_kwargs)
        return cls(model)

    def generate(
        self,
        prompt_text: str,
        temperature: float | None = None
    ) -> str | None:
        """
        Generate a response from the LLM.

        Args:
            prompt_text: The prompt text to send to the LLM
            temperature: Optional temperature override

        Returns:
            LLM response as string
        """
        prompt = Prompt(text=prompt_text)
        response = self.model.generate(prompt, temperature=temperature, use_system_prompt=False)
        assert not isinstance(response, dict)
        return response

    def get_numeric_rating(
        self,
        prompt_text: str,
        scale: tuple[int, int] = (1, 5),
        default: float | None = None
    ) -> float:
        """
        Get a numeric rating from the LLM.

        Useful for rating/scoring tasks where you expect a number.
        The prompt should instruct the LLM to output only a number.

        Args:
            prompt_text: Prompt asking for a numeric rating
            scale: (min, max) values expected
            default: Default value if parsing fails (defaults to middle of scale)

        Returns:
            Numeric rating within the scale
        """
        response = self.generate(prompt_text, temperature=0.1)

        try:
            # Extract first number in the scale range
            pattern = f'[{scale[0]}-{scale[1]}]'
            match = re.search(pattern, str(response))
            if match:
                return float(match.group())
            else:
                if default is None:
                    default = (scale[0] + scale[1]) / 2
                return default
        except Exception:
            if default is None:
                default = (scale[0] + scale[1]) / 2
            return default

    def classify(
        self,
        prompt_text: str,
        options: list[str],
        temperature: float = 0.1,
        case_sensitive: bool = False
    ) -> str | None:
        """
        Classify text using the LLM.

        Useful for binary or multi-class classification tasks.
        The prompt should instruct the LLM to choose from the options.

        Args:
            prompt_text: Prompt asking for classification
            options: List of valid classification options
            temperature: Temperature for generation
            case_sensitive: Whether to match options case-sensitively

        Returns:
            One of the options, or None if no match found
        """
        response = self.generate(prompt_text, temperature=temperature)
        response_str = str(response)

        if not case_sensitive:
            response_str = response_str.lower()
            options = [opt.lower() for opt in options]

        # Find first matching option
        for option in options:
            if option in response_str:
                return option

        return None

    def extract_json(
        self,
        prompt_text: str,
        temperature: float = 0.3
    ) -> dict | list | None:
        """
        Extract JSON from LLM response.

        Useful for structured output tasks.
        The prompt should instruct the LLM to output JSON.

        Args:
            prompt_text: Prompt asking for JSON output
            temperature: Temperature for generation

        Returns:
            Parsed JSON (dict or list), or None if parsing fails
        """
        response = self.generate(prompt_text, temperature=temperature)

        try:
            import json
            import re

            # Try to extract JSON from response (handles markdown code blocks)
            json_match = re.search(r'[\{\[].*[\}\]]', str(response), re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return None
        except Exception:
            return None

    def get_model_stats(self) -> dict:
        """
        Get statistics about model usage (calls, tokens, costs).

        Returns:
            Dictionary with model usage statistics
        """
        return self.model.get_stats()
