"""Data models for Source Utilization and Tool Effectiveness Analyzer."""

from pydantic import BaseModel, Field


# ============================================================================
# Per-Iteration Extracted Data
# ============================================================================


class IterationUtilization(BaseModel):
    """Source utilization data for a single iteration."""

    iteration_num: int
    total_sources: int
    unique_sources: int
    useful_sources: int

    @property
    def utilization_rate(self) -> float:
        """Utilization rate = useful sources / total sources."""
        return self.useful_sources / self.total_sources if self.total_sources > 0 else 0.0


class ToolExecution(BaseModel):
    """Tool execution result."""

    tool_name: str
    query: str
    success: bool  # False if error occurred
    is_none_response: bool  # True if evidence summary is "NONE"
    is_empty_result: bool  # True if no sources returned


class CrossReference(BaseModel):
    """Cross-referencing information for corroborating facts."""

    fact_statement: str
    supporting_sources: list[str]  # List of URLs that support this fact

    @property
    def num_supporting_sources(self) -> int:
        """Number of sources supporting this fact."""
        return len(self.supporting_sources)


class IterationToolEffectiveness(BaseModel):
    """Tool effectiveness data for a single iteration."""

    iteration_num: int
    tool_executions: list[ToolExecution]

    @property
    def num_tool_calls(self) -> int:
        """Total number of tool calls."""
        return len(self.tool_executions)

    @property
    def num_successful_tools(self) -> int:
        """Number of successful tool calls."""
        return sum(1 for t in self.tool_executions if t.success)

    @property
    def num_failed_tools(self) -> int:
        """Number of failed tool calls."""
        return sum(1 for t in self.tool_executions if not t.success)

    @property
    def num_none_responses(self) -> int:
        """Number of tool calls that returned NONE."""
        return sum(1 for t in self.tool_executions if t.is_none_response)

    @property
    def num_empty_results(self) -> int:
        """Number of tool calls that returned empty results."""
        return sum(1 for t in self.tool_executions if t.is_empty_result)

    @property
    def tool_success_rate(self) -> float:
        """Tool success rate."""
        return self.num_successful_tools / self.num_tool_calls if self.num_tool_calls > 0 else 0.0


# ============================================================================
# Per-Claim Extracted Data
# ============================================================================


class SourceUtilizationExtractedData(BaseModel):
    """Extracted data for a single claim."""

    claim_id: str
    claim_text: str
    prediction_correct: bool

    # Source utilization metrics (per-iteration)
    iteration_utilization: list[IterationUtilization]

    # These need to be stored because they require cross-iteration deduplication
    overall_unique_sources: int  # Deduplicated across all iterations
    overall_useful_sources: int  # Deduplicated across all iterations

    # Tool effectiveness metrics (per-iteration)
    iteration_tool_effectiveness: list[IterationToolEffectiveness]

    # Cross-referencing analysis
    cross_references: list[CrossReference]

    # Reasoning citation analysis
    reasoning_cites_evidence: bool  # Whether final reasoning section cites evidence
    reasoning_citation_score: float  # 0-1 score of how well evidence is used

    @property
    def overall_total_sources(self) -> int:
        """Total sources across all iterations (with duplicates)."""
        return sum(it.total_sources for it in self.iteration_utilization)

    @property
    def overall_utilization_rate(self) -> float:
        """Utilization rate based on unique sources."""
        return (
            self.overall_useful_sources / self.overall_unique_sources
            if self.overall_unique_sources > 0
            else 0.0
        )

    @property
    def overall_num_tool_calls(self) -> int:
        """Total tool calls across all iterations."""
        return sum(it.num_tool_calls for it in self.iteration_tool_effectiveness)

    @property
    def overall_num_successful_tools(self) -> int:
        """Total successful tool calls across all iterations."""
        return sum(it.num_successful_tools for it in self.iteration_tool_effectiveness)

    @property
    def overall_num_failed_tools(self) -> int:
        """Total failed tool calls across all iterations."""
        return sum(it.num_failed_tools for it in self.iteration_tool_effectiveness)

    @property
    def overall_num_none_responses(self) -> int:
        """Total NONE responses across all iterations."""
        return sum(it.num_none_responses for it in self.iteration_tool_effectiveness)

    @property
    def overall_num_empty_results(self) -> int:
        """Total empty results across all iterations."""
        return sum(it.num_empty_results for it in self.iteration_tool_effectiveness)

    @property
    def overall_tool_success_rate(self) -> float:
        """Overall tool success rate."""
        return (
            self.overall_num_successful_tools / self.overall_num_tool_calls
            if self.overall_num_tool_calls > 0
            else 0.0
        )

    @property
    def overall_none_response_rate(self) -> float:
        """Overall NONE response rate."""
        return (
            self.overall_num_none_responses / self.overall_num_tool_calls
            if self.overall_num_tool_calls > 0
            else 0.0
        )

    @property
    def overall_empty_result_rate(self) -> float:
        """Overall empty result rate."""
        return (
            self.overall_num_empty_results / self.overall_num_tool_calls
            if self.overall_num_tool_calls > 0
            else 0.0
        )

    @property
    def num_facts_with_cross_references(self) -> int:
        """Number of facts with cross-references."""
        return len(self.cross_references)

    @property
    def avg_sources_per_fact(self) -> float:
        """Average number of sources supporting each cross-referenced fact."""
        if not self.cross_references:
            return 0.0
        return sum(cr.num_supporting_sources for cr in self.cross_references) / len(
            self.cross_references
        )


# ============================================================================
# Aggregated Metrics
# ============================================================================


class UtilizationMetrics(BaseModel):
    """Source utilization metrics."""

    avg_utilization_rate: float
    avg_total_sources: float
    avg_useful_sources: float
    avg_unique_sources: float
    utilization_rate_by_iteration: list[float]  # Average per iteration position


class ToolEffectivenessMetrics(BaseModel):
    """Tool execution effectiveness metrics."""

    avg_tool_success_rate: float
    avg_none_response_rate: float
    avg_empty_result_rate: float
    total_tool_calls: int
    total_successful_tools: int
    total_failed_tools: int
    total_none_responses: int
    total_empty_results: int


class CrossReferencingMetrics(BaseModel):
    """Cross-referencing metrics."""

    avg_num_cross_referenced_facts: float
    avg_sources_per_fact: float
    pct_claims_with_cross_references: float  # Percentage of claims with any cross-references


class ReasoningCitationMetrics(BaseModel):
    """Reasoning citation metrics."""

    avg_reasoning_citation_score: float
    pct_reasoning_cites_evidence: float  # Percentage of claims where reasoning cites evidence


class CorrelationMetrics(BaseModel):
    """Correlation with success metrics."""

    corr_utilization_rate_success: tuple[float, float] | None  # (r, p-value)
    corr_tool_success_rate_success: tuple[float, float] | None
    corr_cross_referencing_success: tuple[float, float] | None
    corr_reasoning_citation_success: tuple[float, float] | None


class FailedVsSuccessfulComparison(BaseModel):
    """Comparison between failed and successful cases."""

    failed_avg_utilization_rate: float
    successful_avg_utilization_rate: float
    failed_avg_tool_success_rate: float
    successful_avg_tool_success_rate: float
    failed_avg_none_response_rate: float
    successful_avg_none_response_rate: float
    failed_avg_cross_referenced_facts: float
    successful_avg_cross_referenced_facts: float
    failed_avg_reasoning_citation_score: float
    successful_avg_reasoning_citation_score: float


class SourceUtilizationMetrics(BaseModel):
    """Complete metrics for source utilization and tool effectiveness analysis."""

    utilization: UtilizationMetrics
    tool_effectiveness: ToolEffectivenessMetrics
    cross_referencing: CrossReferencingMetrics
    reasoning_citation: ReasoningCitationMetrics
    correlations: CorrelationMetrics
    failed_vs_successful: FailedVsSuccessfulComparison


# ============================================================================
# Insights
# ============================================================================


class SourceUtilizationInsights(BaseModel):
    """Human-readable insights from source utilization analysis."""

    # Summary
    summary: str = Field(
        description="High-level summary of source utilization and tool effectiveness"
    )

    # Source utilization insights
    utilization_patterns: str = Field(
        description="Patterns in how sources are utilized (found vs marked useful)"
    )

    # Tool effectiveness insights
    tool_effectiveness_patterns: str = Field(
        description="Patterns in tool execution success/failure and information loss"
    )

    # Cross-referencing insights
    cross_referencing_patterns: str = Field(
        description="Patterns in cross-referencing and corroboration"
    )

    # Reasoning citation insights
    reasoning_citation_patterns: str = Field(
        description="Patterns in how evidence informs reasoning"
    )

    # Success factors
    success_factors: str = Field(
        description="What distinguishes successful from failed fact-checking attempts"
    )

    # Recommendations
    recommendations: str = Field(description="Actionable recommendations for improvement")
