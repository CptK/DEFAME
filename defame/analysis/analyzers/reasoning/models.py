"""Data models for Reasoning Quality Analyzer."""

from pydantic import BaseModel, Field


# ============================================================================
# Per-Iteration Extracted Data
# ============================================================================


class LogicalFallacy(BaseModel):
    """A logical fallacy identified in reasoning."""

    fallacy_type: str  # e.g., "circular reasoning", "false dichotomy"
    description: str  # Brief explanation of the fallacy instance


class ReasoningBlock(BaseModel):
    """Reasoning extracted from a single iteration."""

    iteration_num: int
    reasoning_text: str  # The raw reasoning text from elaboration


class IterationReasoningQuality(BaseModel):
    """Reasoning quality assessment for a single iteration."""

    iteration_num: int
    has_reasoning: bool  # Whether reasoning exists for this iteration

    # Logical coherence (1-5 scale)
    logical_coherence_score: float  # 0.0 if no reasoning
    coherence_explanation: str  # LLM explanation of coherence rating

    # Evidence-to-claim connection
    addresses_claim: bool  # Whether reasoning addresses the claim
    evidence_claim_strength: float  # 0-1 score of connection strength
    connection_explanation: str  # LLM explanation

    # Logical chain analysis
    logical_chain_strength: float  # 1-5 score of Evidence → Inference → Conclusion flow
    chain_breaks: list[str]  # List of identified breaks/gaps in logical chain
    chain_explanation: str  # LLM explanation of chain analysis

    # Logical fallacies
    logical_fallacies: list[LogicalFallacy]

    # Synthesis quality (across sources)
    synthesis_quality_score: float  # 1-5 score of how well sources are synthesized
    synthesis_explanation: str  # LLM explanation


# ============================================================================
# Per-Claim Extracted Data
# ============================================================================


class ReasoningExtractedData(BaseModel):
    """Extracted reasoning quality data for a single claim."""

    claim_id: str
    claim_text: str
    prediction_correct: bool

    # Per-iteration reasoning quality
    iteration_reasoning: list[IterationReasoningQuality]
    correct: bool

    @property
    def num_iterations_with_reasoning(self) -> int:
        """Number of iterations that have reasoning."""
        return sum(1 for it in self.iteration_reasoning if it.has_reasoning)

    @property
    def avg_logical_coherence(self) -> float:
        """Average logical coherence score across iterations with reasoning."""
        scores = [it.logical_coherence_score for it in self.iteration_reasoning if it.has_reasoning]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def avg_evidence_claim_strength(self) -> float:
        """Average evidence-claim connection strength."""
        scores = [it.evidence_claim_strength for it in self.iteration_reasoning if it.has_reasoning]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def avg_chain_strength(self) -> float:
        """Average logical chain strength."""
        scores = [it.logical_chain_strength for it in self.iteration_reasoning if it.has_reasoning]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def avg_synthesis_quality(self) -> float:
        """Average synthesis quality score."""
        scores = [it.synthesis_quality_score for it in self.iteration_reasoning if it.has_reasoning]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def total_logical_fallacies(self) -> int:
        """Total number of logical fallacies across all iterations."""
        return sum(len(it.logical_fallacies) for it in self.iteration_reasoning)

    @property
    def pct_iterations_address_claim(self) -> float:
        """Percentage of iterations where reasoning addresses the claim."""
        if not self.num_iterations_with_reasoning:
            return 0.0
        count = sum(1 for it in self.iteration_reasoning if it.has_reasoning and it.addresses_claim)
        return count / self.num_iterations_with_reasoning

    @property
    def total_chain_breaks(self) -> int:
        """Total number of logical chain breaks."""
        return sum(len(it.chain_breaks) for it in self.iteration_reasoning)


# ============================================================================
# Aggregated Metrics
# ============================================================================


class CoherenceMetrics(BaseModel):
    """Logical coherence metrics."""

    avg_coherence_score: float  # Average across all claims
    coherence_score_by_iteration: list[float]  # Average per iteration position
    pct_high_coherence: float  # Percentage with score >= 4


class EvidenceClaimMetrics(BaseModel):
    """Evidence-to-claim connection metrics."""

    avg_evidence_claim_strength: float
    pct_addresses_claim: float  # Percentage where reasoning addresses claim
    pct_strong_connection: float  # Percentage with strength >= 0.7


class LogicalChainMetrics(BaseModel):
    """Logical chain strength metrics."""

    avg_chain_strength: float
    avg_chain_breaks_per_claim: float
    pct_strong_chains: float  # Percentage with chain strength >= 4
    common_chain_breaks: list[tuple[str, int]]  # (break type, count) - top 5


class SynthesisMetrics(BaseModel):
    """Synthesis quality metrics."""

    avg_synthesis_quality: float
    synthesis_quality_by_iteration: list[float]
    pct_good_synthesis: float  # Percentage with score >= 4


class FallacyMetrics(BaseModel):
    """Logical fallacy metrics."""

    avg_fallacies_per_claim: float
    total_fallacies: int
    common_fallacies: list[tuple[str, int]]  # (fallacy type, count) - top 5
    pct_claims_with_fallacies: float


class ReasoningCorrelationMetrics(BaseModel):
    """Correlation with success metrics."""

    corr_coherence_success: tuple[float, float] | None  # (r, p-value)
    corr_evidence_claim_strength_success: tuple[float, float] | None
    corr_chain_strength_success: tuple[float, float] | None
    corr_synthesis_quality_success: tuple[float, float] | None
    corr_fallacy_count_success: tuple[float, float] | None


class FailedVsSuccessfulReasoningComparison(BaseModel):
    """Comparison between failed and successful cases."""

    failed_avg_coherence: float
    successful_avg_coherence: float
    failed_avg_evidence_claim_strength: float
    successful_avg_evidence_claim_strength: float
    failed_avg_chain_strength: float
    successful_avg_chain_strength: float
    failed_avg_synthesis_quality: float
    successful_avg_synthesis_quality: float
    failed_avg_fallacies: float
    successful_avg_fallacies: float


class ReasoningMetrics(BaseModel):
    """Complete metrics for reasoning quality analysis."""

    coherence: CoherenceMetrics
    evidence_claim: EvidenceClaimMetrics
    logical_chain: LogicalChainMetrics
    synthesis: SynthesisMetrics
    fallacies: FallacyMetrics
    correlations: ReasoningCorrelationMetrics
    failed_vs_successful: FailedVsSuccessfulReasoningComparison


# ============================================================================
# Insights
# ============================================================================


class ReasoningInsights(BaseModel):
    """Human-readable insights from reasoning quality analysis."""

    summary: str = Field(
        description="High-level summary of reasoning quality patterns"
    )

    coherence_patterns: str = Field(
        description="Patterns in logical coherence across claims"
    )

    evidence_integration_patterns: str = Field(
        description="Patterns in how evidence connects to claims"
    )

    logical_chain_patterns: str = Field(
        description="Patterns in logical flow from evidence to conclusions"
    )

    synthesis_patterns: str = Field(
        description="Patterns in synthesis quality across sources"
    )

    fallacy_patterns: str = Field(
        description="Common logical fallacies and their impact"
    )

    failure_drivers: str = Field(
        description="Analysis of whether weak reasoning drives failure"
    )

    success_factors: str = Field(
        description="Reasoning characteristics that distinguish successful cases"
    )

    recommendations: str = Field(
        description="Actionable recommendations for improving reasoning quality"
    )
