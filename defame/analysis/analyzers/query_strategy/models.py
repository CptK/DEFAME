from pydantic import BaseModel, Field
from typing import Any


# ============================================================================
# Extracted Data Models (Per Iteration / Per Claim)
# ============================================================================

class QueryInfo(BaseModel):
    """Information about a single query."""
    query_text: str
    tool_name: str  # e.g., "web_search", "image_search"
    specificity_score: float | None  # How specific this query is (e.g., length, entity count, IDF)
    keyword_overlap_with_claim: float | None  # Cosine similarity or overlap with claim
    is_counter_evidence_seeking: bool | None  # Whether this query seeks counter-evidence


class ExtractedIterationData(BaseModel):
    """Data for a single iteration regarding query strategies."""
    iteration_number: int
    queries: list[QueryInfo]  # All queries made in this iteration
    tool_types_used: list[str]  # Distinct tool types used (e.g., ["web_search", "image_search"])

    @property
    def num_queries(self) -> int:
        return len(self.queries)

    @property
    def avg_specificity(self) -> float:
        """Average specificity score, ignoring None values."""
        if not self.queries:
            return 0.0
        scores = [q.specificity_score for q in self.queries if q.specificity_score is not None]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def avg_keyword_overlap(self) -> float:
        """Average keyword overlap, ignoring None values."""
        if not self.queries:
            return 0.0
        overlaps = [q.keyword_overlap_with_claim for q in self.queries if q.keyword_overlap_with_claim is not None]
        return sum(overlaps) / len(overlaps) if overlaps else 0.0

    @property
    def num_specificity_scores_available(self) -> int:
        """Count how many queries have specificity scores (for data quality tracking)."""
        return sum(1 for q in self.queries if q.specificity_score is not None)

    @property
    def num_keyword_overlap_scores_available(self) -> int:
        """Count how many queries have keyword overlap scores (for data quality tracking)."""
        return sum(1 for q in self.queries if q.keyword_overlap_with_claim is not None)


class QueryEvolution(BaseModel):
    """Tracks how queries evolve across iterations."""
    specificity_trend: list[float] = Field(..., description="Avg specificity per iteration")
    keyword_overlap_trend: list[float]  # Avg keyword overlap per iteration
    lexical_diversity_between_iterations: list[float]  # Diversity score between consecutive iterations (length = num_iterations - 1)
    query_count_per_iteration: list[int]  # Number of queries per iteration


class ToolDiversity(BaseModel):
    """Tracks tool usage diversity and redundancy."""
    unique_tool_types: list[str]  # List of unique tool types used
    tool_type_counts: dict[str, int]  # Count of each tool type
    num_redundant_queries: int  # Number of queries that are similar/duplicate to previous ones
    total_queries: int  # Total number of queries

    @property
    def diversity_score(self) -> float:
        """Ratio of unique tool types to total queries."""
        if self.total_queries == 0:
            return 0.0
        return len(self.unique_tool_types) / self.total_queries

    @property
    def redundancy_ratio(self) -> float:
        """Ratio of redundant queries to total queries."""
        if self.total_queries == 0:
            return 0.0
        return self.num_redundant_queries / self.total_queries


class ToolChoiceCoherence(BaseModel):
    """Tracks whether tool choices are informed by previous results."""
    # For each action (after the first iteration), did it logically follow from previous evidence?
    coherence_scores: list[float]  # Scores from 0-1 for each action (except first iteration)

    @property
    def avg_coherence(self) -> float:
        """Average coherence score."""
        if not self.coherence_scores:
            return 0.0
        return sum(self.coherence_scores) / len(self.coherence_scores)


class CounterEvidenceSeeking(BaseModel):
    """Tracks evidence of counter-evidence seeking behavior."""
    num_counter_evidence_queries: int  # Number of queries seeking counter-evidence
    total_queries: int  # Total number of queries
    counter_evidence_queries: list[str]  # Actual queries that seek counter-evidence

    @property
    def counter_evidence_ratio(self) -> float:
        """Ratio of counter-evidence seeking queries to total queries."""
        if self.total_queries == 0:
            return 0.0
        return self.num_counter_evidence_queries / self.total_queries


class SearchAngleAnalysis(BaseModel):
    """Analysis of different search angles/topics covered."""
    num_distinct_angles: int  # Number of distinct topics/aspects covered
    angle_groups: list[list[str]]  # Grouped queries by topic (each group is a list of query texts)
    angle_labels: list[str]  # Labels for each angle/topic


class QueryStrategyExtractedData(BaseModel):
    """Extracted data for a single claim log regarding query strategies."""
    claim_id: int
    success: bool  # Whether the claim was predicted correctly
    claim_text: str  # The actual claim text (for overlap calculations)

    iterations: list[ExtractedIterationData]  # Data for each iteration

    # Aggregate metrics for this claim
    query_evolution: QueryEvolution
    tool_diversity: ToolDiversity
    tool_choice_coherence: ToolChoiceCoherence
    counter_evidence_seeking: CounterEvidenceSeeking
    search_angle_analysis: SearchAngleAnalysis


# ============================================================================
# Metrics Models (Aggregated Across All Claims)
# ============================================================================

class QueryStrategyMetrics(BaseModel):
    """Metrics computed across all claim logs regarding query strategies."""
    # Basic query statistics
    avg_num_queries_per_iteration: list[float]  # Average number of queries per iteration (index 0 = iteration 1)
    avg_query_specificity: float  # Average query specificity across all iterations and claims
    avg_query_specificity_per_iteration: list[float]  # Average query specificity per iteration
    avg_keyword_overlap_with_claim: float  # Average keyword overlap with claim across all
    avg_keyword_overlap_per_iteration: list[float]  # Average keyword overlap per iteration

    # Search angle diversity
    avg_num_search_angles_used: float  # Average number of distinct search angles per claim

    # Query evolution metrics
    avg_specificity_change_over_iterations: float  # Average change in specificity from first to last iteration
    avg_lexical_diversity_between_iterations: float  # Average lexical diversity between consecutive iterations

    # Tool diversity metrics
    avg_tool_diversity_score: float  # Average tool diversity score
    avg_redundancy_ratio: float  # Average query redundancy ratio
    most_common_tools: list[tuple[str, int]]  # Most commonly used tools and their counts

    # Tool choice coherence metrics
    avg_tool_choice_coherence: float  # Average coherence score

    # Counter-evidence seeking metrics
    avg_counter_evidence_ratio: float  # Average ratio of counter-evidence seeking queries

    # Correlation with success
    corr_num_queries_success: tuple[float, float]  # (correlation, p-value)
    corr_specificity_success: tuple[float, float]
    corr_tool_diversity_success: tuple[float, float]
    corr_counter_evidence_success: tuple[float, float]

    # Comparison: failed vs successful cases
    failed_vs_successful_comparison: dict[str, Any] = Field(
        default_factory=dict,
        description="Comparison of metrics between failed and successful cases"
    )


# ============================================================================
# Insights Models
# ============================================================================

class QueryStrategyInsights(BaseModel):
    """High-level insights regarding query strategies."""
    # Overall summary
    insights_summary: str  # Summary of all insights

    # Specific insights for key questions
    query_vagueness_in_failures: str  # Are failed cases more vague in queries?
    query_repetition_patterns: str  # Do they repeat the same query phrasing?
    specificity_evolution: str  # Do queries get more/less specific over iterations?
    tool_diversity_insights: str  # Insights about tool usage diversity
    counter_evidence_insights: str  # Insights about counter-evidence seeking behavior

    # Correlation insights
    correlation_insights: str  # Insights from correlation analysis
