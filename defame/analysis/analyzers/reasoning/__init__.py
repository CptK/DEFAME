"""Reasoning Quality Analyzer.

This module analyzes the quality of reasoning in fact-checking attempts,
including logical coherence, evidence-to-claim connections, logical chain
strength, synthesis quality, and logical fallacies.
"""

from defame.analysis.analyzers.reasoning.data_extraction import ReasoningExtractor
from defame.analysis.analyzers.reasoning.models import (
    CoherenceMetrics,
    EvidenceClaimMetrics,
    FailedVsSuccessfulReasoningComparison,
    FallacyMetrics,
    IterationReasoningQuality,
    LogicalChainMetrics,
    LogicalFallacy,
    ReasoningBlock,
    ReasoningCorrelationMetrics,
    ReasoningExtractedData,
    ReasoningInsights,
    ReasoningMetrics,
    SynthesisMetrics,
)
from defame.analysis.analyzers.reasoning.reasoning_analyzer import ReasoningAnalyzer

__all__ = [
    # Main analyzer
    "ReasoningAnalyzer",
    # Data extraction
    "ReasoningExtractor",
    # Models - extracted data
    "ReasoningBlock",
    "LogicalFallacy",
    "IterationReasoningQuality",
    "ReasoningExtractedData",
    # Models - metrics
    "CoherenceMetrics",
    "EvidenceClaimMetrics",
    "LogicalChainMetrics",
    "SynthesisMetrics",
    "FallacyMetrics",
    "ReasoningCorrelationMetrics",
    "FailedVsSuccessfulReasoningComparison",
    "ReasoningMetrics",
    # Models - insights
    "ReasoningInsights",
]
