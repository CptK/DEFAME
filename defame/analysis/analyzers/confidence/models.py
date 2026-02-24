"""Data models for Confidence Analyzer."""

from pydantic import BaseModel, Field


# ============================================================================
# Per-Iteration Extracted Data
# ============================================================================

class ConfidenceStrength(BaseModel):
    """Confidence strength assessment for a single iteration."""

    iteration_num: int
    confidence_strength: float  # 1-5 score of confidence in the claim based on evidence
    evidence_strength: float  # 1-5 score of evidence quality
    hedging_score: float  # 1-5 score indicating hedging language (1=no hedging, 5=heavy hedging)


# ============================================================================
# Per-Claim Extracted Data
# ============================================================================

class ConfidenceExtractedData(BaseModel):
    """Extracted confidence data for a single claim."""

    claim_id: str
    claim_text: str
    prediction_correct: bool
    iterations: list[ConfidenceStrength] = Field(
        default_factory=list,
        description="List of confidence strength assessments per iteration."
    )
    correct: bool

    @property
    def avg_confidence_strength(self) -> float:
        """Average confidence strength across iterations."""
        if not self.iterations:
            return 0.0
        total = sum(it.confidence_strength for it in self.iterations)
        return total / len(self.iterations)
    
    @property
    def avg_evidence_strength(self) -> float:
        """Average evidence strength across iterations."""
        if not self.iterations:
            return 0.0
        total = sum(it.evidence_strength for it in self.iterations)
        return total / len(self.iterations)

    @property
    def avg_hedging_score(self) -> float:
        """Average hedging score across iterations."""
        if not self.iterations:
            return 0.0
        total = sum(it.hedging_score for it in self.iterations)
        return total / len(self.iterations)


# ============================================================================
# Aggregated Metrics
# ============================================================================

class ConfidenceMetrics(BaseModel):
    """Aggregated confidence metrics across all claims."""

    avg_confidence_strength: float
    avg_evidence_strength: float
    corr_confidence_evidence: tuple[float, float] | None  # (r, p-value)  # Correlation between confidence and evidence strength
    corr_confidence_accuracy: tuple[float, float] | None  # (r, p-value)  # Correlation between confidence and prediction correctness
    # Confidence pattern in wrong predictions
    avg_confidence_wrong: float
    avg_evidence_wrong: float
    # Confidence pattern in correct predictions
    avg_confidence_correct: float
    avg_evidence_correct: float
    # Hedging metrics
    avg_hedging_score: float
    avg_hedging_wrong: float
    avg_hedging_correct: float
    corr_hedging_accuracy: tuple[float, float] | None  # (r, p-value)  # Correlation between hedging and correctness
    # Within-claim confidence evolution (tracks same claims over iterations)
    avg_confidence_change_per_claim: float  # Average change from first to last iteration within claims
    pct_claims_confidence_increased: float  # Percentage of multi-iteration claims where confidence increased
    pct_claims_confidence_decreased: float  # Percentage of multi-iteration claims where confidence decreased

    @property
    def confidence_accuracy_gap(self) -> float:
        """Gap in average confidence between correct and incorrect predictions."""
        return self.avg_confidence_correct - self.avg_confidence_wrong

    @property
    def hedging_accuracy_gap(self) -> float:
        """Gap in average hedging between correct and incorrect predictions."""
        return self.avg_hedging_correct - self.avg_hedging_wrong


# ============================================================================
# Insights
# ============================================================================

class ConfidenceInsights(BaseModel):
    """Insights derived from confidence analysis."""

    overconfidence_in_failures: str = Field(
        ..., description="Whether incorrect predictions show overconfidence."
    )
    confidence_threshold_for_failure: str = Field(
        ..., description="Confidence score threshold below which failures are more likely."
    )
    hedging_correlation_with_correctness: str = Field(
        ..., description="Correlation between hedging behavior and correctness."
    )
    confidence_evolution_pattern: str = Field(
        ..., description="How confidence evolves within individual claims across iterations."
    )
