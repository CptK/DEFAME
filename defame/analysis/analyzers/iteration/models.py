"""Data models for Iteration & Stopping Criteria Analyzer."""

from pydantic import BaseModel, Field


# ============================================================================
# Per-Iteration Extracted Data
# ============================================================================

class IterationInfo(BaseModel):
    """Information about a single iteration."""

    iteration_num: int
    has_judgement: bool  # Whether this iteration produced a judgement
    information_gain: float  # 1-5 score of new information discovered (0 if no evidence)
    num_evidence_pieces: int  # Number of evidence pieces in this iteration


# ============================================================================
# Per-Claim Extracted Data
# ============================================================================

class StoppingDecision(BaseModel):
    """Assessment of whether stopping was appropriate."""

    decision: str  # "Appropriate", "Should_continue", "Uncertain"
    reasoning: str  # LLM explanation


class IterationExtractedData(BaseModel):
    """Extracted iteration data for a single claim."""

    claim_id: int
    claim_text: str
    prediction_correct: bool
    num_iterations: int
    max_iterations_available: int  # Maximum iterations the system could have used
    iterations: list[IterationInfo] = Field(default_factory=list)
    stopping_decision: StoppingDecision | None = None

    @property
    def avg_information_gain(self) -> float:
        """Average information gain across iterations with evidence."""
        gains = [it.information_gain for it in self.iterations if it.information_gain > 0.0]
        return sum(gains) / len(gains) if gains else 0.0

    @property
    def total_evidence_pieces(self) -> int:
        """Total evidence pieces collected across all iterations."""
        return sum(it.num_evidence_pieces for it in self.iterations)


# ============================================================================
# Aggregated Metrics
# ============================================================================

class IterationMetrics(BaseModel):
    """Aggregated iteration metrics across all claims."""

    # Iteration usage
    avg_num_iterations: float
    avg_num_iterations_wrong: float
    avg_num_iterations_correct: float
    pct_using_max_iterations: float  # Percentage of claims that hit max iterations

    # Information gain
    avg_information_gain_per_iteration: float
    avg_total_information_gain: float  # Sum across all iterations per claim
    information_gain_by_iteration: list[float]  # Average gain at each iteration number

    # Evidence accumulation
    avg_evidence_per_iteration: float
    avg_total_evidence_per_claim: float
    evidence_by_iteration: list[float]  # Average evidence count at each iteration

    # Stopping decisions
    pct_appropriate_stopping: float
    pct_should_continue: float
    pct_uncertain_stopping: float

    # Early stopping in failures
    pct_failures_stopped_early: float  # Failures that didn't use max iterations
    pct_successes_stopped_early: float

    @property
    def iteration_gap(self) -> float:
        """Difference in iterations between correct and incorrect predictions."""
        return self.avg_num_iterations_correct - self.avg_num_iterations_wrong


# ============================================================================
# Insights
# ============================================================================

class IterationInsights(BaseModel):
    """Insights derived from iteration analysis."""

    failures_use_fewer_iterations: str = Field(
        ..., description="Whether failures tend to use fewer iterations than successes."
    )
    early_stopping_in_failures: str = Field(
        ..., description="Whether failures stop too early before finding key evidence."
    )
    information_gain_pattern: str = Field(
        ..., description="How information gain evolves across iterations (plateau detection)."
    )
    stopping_decision_quality: str = Field(
        ..., description="Overall quality of stopping decisions."
    )
