"""
Configuration models for analyzers.

Provides type-safe configuration using Pydantic models.
"""

from pydantic import BaseModel, Field
from typing import Any

from defame.analysis.llm_helper import AnalyzerLLMHelper


class AnalyzerConfig(BaseModel):
    """
    Base configuration for all analyzers.

    This provides common configuration options that all analyzers can use.
    Subclass this for analyzer-specific config options.
    """
    model_config = {"arbitrary_types_allowed": True}

    # LLM configuration
    llm: AnalyzerLLMHelper | None = Field(
        default=None,
        description="LLM helper instance for analyzers that use LLMs"
    )

    # Processing configuration
    use_multiprocessing: bool = Field(
        default=True,
        description="Whether to use multiprocessing for parallel processing of claims"
    )

    # Additional metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional analyzer-specific metadata"
    )
