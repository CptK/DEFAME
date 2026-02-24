"""Source Utilization and Tool Effectiveness Analyzer."""

from __future__ import annotations

from multiprocessing import Pool

import numpy as np
from scipy import stats
from typing import cast
from tqdm import tqdm

from defame.analysis.analyzer_config import AnalyzerConfig
from defame.analysis.analyzers.base_analyzer import BaseAnalyzer
from defame.analysis.analyzers.source_utilization.data_extraction import (
    SourceUtilizationExtractor,
)
from defame.analysis.analyzers.source_utilization.models import (
    CorrelationMetrics,
    CrossReferencingMetrics,
    FailedVsSuccessfulComparison,
    ReasoningCitationMetrics,
    SourceUtilizationExtractedData,
    SourceUtilizationInsights,
    SourceUtilizationMetrics,
    ToolEffectivenessMetrics,
    UtilizationMetrics,
)
from defame.analysis.data_models import ClaimData, ExperimentData
from defame.analysis.llm_helper import AnalyzerLLMHelper


def _process_claim_log_wrapper(
    args: tuple[dict, ClaimData]
) -> SourceUtilizationExtractedData | None:
    """
    Wrapper function for multiprocessing to process a single claim.

    Args:
        args: Tuple of (config_dict, claim_data)

    Returns:
        SourceUtilizationExtractedData or None if processing fails
    """
    config_dict, claim_data = args

    try:
        # Reconstruct LLM helper in worker process
        from defame.common.modeling import make_model

        llm_helper = None
        if config_dict.get("llm_config"):
            llm_config = config_dict["llm_config"]
            # Create model with temperature parameter
            model_kwargs = {"temperature": llm_config["temperature"]}
            model = make_model(llm_config["model_name"], **model_kwargs)
            llm_helper = AnalyzerLLMHelper.from_model(model)

        extractor = SourceUtilizationExtractor(
            llm_helper=llm_helper,
            use_embedding_clustering=config_dict.get("use_embedding_clustering", True),
        )

        return extractor.process_claim_data(claim_data)

    except Exception as e:
        print(f"Error processing claim {claim_data.claim_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


class SourceUtilizationAnalyzer(
    BaseAnalyzer[
        SourceUtilizationExtractedData,
        SourceUtilizationMetrics,
        SourceUtilizationInsights,
    ]
):
    """Analyzer for source utilization and tool effectiveness."""

    def __init__(
        self, config: AnalyzerConfig, use_embedding_clustering: bool = True
    ) -> None:
        """
        Initialize the analyzer.

        Args:
            config: Analyzer configuration
            use_embedding_clustering: Whether to use embedding-based cross-ref clustering
        """
        super().__init__(config)
        self.use_embedding_clustering = use_embedding_clustering

    def extract_data(
        self, experiment_data: ExperimentData
    ) -> list[SourceUtilizationExtractedData]:
        """
        Extract source utilization data from all claims.

        Args:
            experiment_data: Experiment data containing claims

        Returns:
            List of extracted data (one per claim)
        """
        # Serialize config to dict (only serializable data, no LLM helper object)
        config_dict = {
            "llm_config": None,
            "use_embedding_clustering": self.use_embedding_clustering,
        }

        # If LLM is configured, save its initialization parameters
        if self.config.llm:
            config_dict["llm_config"] = {
                "model_name": self.config.llm.model.name,
                "temperature": self.config.llm.model.temperature,
            }

        if self.config.use_multiprocessing:
            # Use multiprocessing
            with Pool() as pool:
                args_list = [
                    (config_dict, claim_data)
                    for claim_data in experiment_data.claims
                ]
                results = list(
                    pool.imap_unordered(_process_claim_log_wrapper, args_list)
                )
                # Filter out None results
                extracted_data = [r for r in results if r is not None]
        else:
            # Single-threaded processing
            extractor = SourceUtilizationExtractor(
                llm_helper=self.config.llm,
                use_embedding_clustering=self.use_embedding_clustering,
            )
            extracted_data = []
            for claim_data in tqdm(experiment_data.claims, desc="Source Utilization", unit="claim"):
                try:
                    data = extractor.process_claim_data(claim_data)
                    extracted_data.append(data)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Error processing claim {claim_data.claim_id}: {e}")

        return extracted_data

    def compute_metrics(
        self, extracted_data: list[SourceUtilizationExtractedData]
    ) -> SourceUtilizationMetrics:
        """
        Compute aggregated metrics from extracted data.

        Args:
            extracted_data: List of extracted data per claim

        Returns:
            Aggregated metrics
        """
        if not extracted_data:
            # Return empty metrics if no data
            return self._empty_metrics()

        # Separate by success/failure
        failed_data = [d for d in extracted_data if not d.prediction_correct]
        successful_data = [d for d in extracted_data if d.prediction_correct]

        # === UTILIZATION METRICS ===
        utilization_rates = [d.overall_utilization_rate for d in extracted_data]
        avg_utilization_rate = float(np.mean(utilization_rates))

        avg_total_sources = float(
            np.mean([d.overall_total_sources for d in extracted_data])
        )
        avg_useful_sources = float(
            np.mean([d.overall_useful_sources for d in extracted_data])
        )
        avg_unique_sources = float(
            np.mean([d.overall_unique_sources for d in extracted_data])
        )

        # Compute per-iteration averages
        max_iterations = max(
            len(d.iteration_utilization) for d in extracted_data if d.iteration_utilization
        )
        utilization_rate_by_iteration = []
        for i in range(max_iterations):
            rates = [
                d.iteration_utilization[i].utilization_rate
                for d in extracted_data
                if i < len(d.iteration_utilization)
            ]
            if rates:
                utilization_rate_by_iteration.append(float(np.mean(rates)))

        utilization = UtilizationMetrics(
            avg_utilization_rate=avg_utilization_rate,
            avg_total_sources=avg_total_sources,
            avg_useful_sources=avg_useful_sources,
            avg_unique_sources=avg_unique_sources,
            utilization_rate_by_iteration=utilization_rate_by_iteration,
        )

        # === TOOL EFFECTIVENESS METRICS ===
        tool_success_rates = [d.overall_tool_success_rate for d in extracted_data]
        avg_tool_success_rate = float(np.mean(tool_success_rates))

        none_response_rates = [d.overall_none_response_rate for d in extracted_data]
        avg_none_response_rate = float(np.mean(none_response_rates))

        empty_result_rates = [d.overall_empty_result_rate for d in extracted_data]
        avg_empty_result_rate = float(np.mean(empty_result_rates))

        total_tool_calls = sum(d.overall_num_tool_calls for d in extracted_data)
        total_successful_tools = sum(
            d.overall_num_successful_tools for d in extracted_data
        )
        total_failed_tools = sum(d.overall_num_failed_tools for d in extracted_data)
        total_none_responses = sum(
            d.overall_num_none_responses for d in extracted_data
        )
        total_empty_results = sum(
            d.overall_num_empty_results for d in extracted_data
        )

        tool_effectiveness = ToolEffectivenessMetrics(
            avg_tool_success_rate=avg_tool_success_rate,
            avg_none_response_rate=avg_none_response_rate,
            avg_empty_result_rate=avg_empty_result_rate,
            total_tool_calls=total_tool_calls,
            total_successful_tools=total_successful_tools,
            total_failed_tools=total_failed_tools,
            total_none_responses=total_none_responses,
            total_empty_results=total_empty_results,
        )

        # === CROSS-REFERENCING METRICS ===
        num_cross_referenced_facts = [
            d.num_facts_with_cross_references for d in extracted_data
        ]
        avg_num_cross_referenced_facts = float(np.mean(num_cross_referenced_facts))

        sources_per_fact = [
            d.avg_sources_per_fact
            for d in extracted_data
            if d.num_facts_with_cross_references > 0
        ]
        avg_sources_per_fact = (
            float(np.mean(sources_per_fact)) if sources_per_fact else 0.0
        )

        pct_claims_with_cross_references = (
            100.0
            * sum(1 for d in extracted_data if d.num_facts_with_cross_references > 0)
            / len(extracted_data)
        )

        cross_referencing = CrossReferencingMetrics(
            avg_num_cross_referenced_facts=avg_num_cross_referenced_facts,
            avg_sources_per_fact=avg_sources_per_fact,
            pct_claims_with_cross_references=pct_claims_with_cross_references,
        )

        # === REASONING CITATION METRICS ===
        citation_scores = [d.reasoning_citation_score for d in extracted_data]
        avg_reasoning_citation_score = float(np.mean(citation_scores))

        pct_reasoning_cites_evidence = (
            100.0
            * sum(1 for d in extracted_data if d.reasoning_cites_evidence)
            / len(extracted_data)
        )

        reasoning_citation = ReasoningCitationMetrics(
            avg_reasoning_citation_score=avg_reasoning_citation_score,
            pct_reasoning_cites_evidence=pct_reasoning_cites_evidence,
        )

        # === CORRELATION METRICS ===
        success_labels = [1 if d.prediction_correct else 0 for d in extracted_data]

        def safe_correlation(values: list[float]) -> tuple[float, float] | None:
            """Compute correlation safely, returning None if invalid."""
            if len(values) < 2 or len(set(success_labels)) < 2:
                return None
            try:
                r, p = stats.pearsonr(values, success_labels)
                assert isinstance(r, float) and isinstance(p, float)

                return (float(r), float(p))
            except Exception:
                return None

        correlations = CorrelationMetrics(
            corr_utilization_rate_success=safe_correlation(utilization_rates),
            corr_tool_success_rate_success=safe_correlation(tool_success_rates),
            corr_cross_referencing_success=safe_correlation(cast(list[float], num_cross_referenced_facts)),
            corr_reasoning_citation_success=safe_correlation(citation_scores),
        )

        # === FAILED VS SUCCESSFUL COMPARISON ===
        def safe_mean(data: list, attr: str) -> float:
            """Safely compute mean of an attribute."""
            if not data:
                return 0.0
            values = [getattr(d, attr) for d in data]
            return float(np.mean(values))

        failed_vs_successful = FailedVsSuccessfulComparison(
            failed_avg_utilization_rate=safe_mean(
                failed_data, "overall_utilization_rate"
            ),
            successful_avg_utilization_rate=safe_mean(
                successful_data, "overall_utilization_rate"
            ),
            failed_avg_tool_success_rate=safe_mean(
                failed_data, "overall_tool_success_rate"
            ),
            successful_avg_tool_success_rate=safe_mean(
                successful_data, "overall_tool_success_rate"
            ),
            failed_avg_none_response_rate=safe_mean(
                failed_data, "overall_none_response_rate"
            ),
            successful_avg_none_response_rate=safe_mean(
                successful_data, "overall_none_response_rate"
            ),
            failed_avg_cross_referenced_facts=safe_mean(
                failed_data, "num_facts_with_cross_references"
            ),
            successful_avg_cross_referenced_facts=safe_mean(
                successful_data, "num_facts_with_cross_references"
            ),
            failed_avg_reasoning_citation_score=safe_mean(
                failed_data, "reasoning_citation_score"
            ),
            successful_avg_reasoning_citation_score=safe_mean(
                successful_data, "reasoning_citation_score"
            ),
        )

        return SourceUtilizationMetrics(
            utilization=utilization,
            tool_effectiveness=tool_effectiveness,
            cross_referencing=cross_referencing,
            reasoning_citation=reasoning_citation,
            correlations=correlations,
            failed_vs_successful=failed_vs_successful,
        )

    def generate_insights(
        self, metrics: SourceUtilizationMetrics
    ) -> SourceUtilizationInsights:
        """
        Generate human-readable insights from metrics.

        Args:
            metrics: Computed metrics

        Returns:
            Structured insights
        """
        # Generate summary
        summary = self._generate_summary(metrics)

        # Utilization patterns
        utilization_patterns = self._generate_utilization_patterns(metrics)

        # Tool effectiveness patterns
        tool_effectiveness_patterns = self._generate_tool_effectiveness_patterns(
            metrics
        )

        # Cross-referencing patterns
        cross_referencing_patterns = self._generate_cross_referencing_patterns(metrics)

        # Reasoning citation patterns
        reasoning_citation_patterns = self._generate_reasoning_citation_patterns(
            metrics
        )

        # Success factors
        success_factors = self._generate_success_factors(metrics)

        # Recommendations
        recommendations = self._generate_recommendations(metrics)

        return SourceUtilizationInsights(
            summary=summary,
            utilization_patterns=utilization_patterns,
            tool_effectiveness_patterns=tool_effectiveness_patterns,
            cross_referencing_patterns=cross_referencing_patterns,
            reasoning_citation_patterns=reasoning_citation_patterns,
            success_factors=success_factors,
            recommendations=recommendations,
        )

    def _generate_summary(self, metrics: SourceUtilizationMetrics) -> str:
        """Generate high-level summary."""
        util = metrics.utilization
        tool = metrics.tool_effectiveness
        cross = metrics.cross_referencing
        cite = metrics.reasoning_citation

        return (
            f"On average, {util.avg_utilization_rate:.1%} of found sources are marked as useful. "
            f"Tool execution succeeds {tool.avg_tool_success_rate:.1%} of the time, "
            f"with {tool.avg_none_response_rate:.1%} NONE responses and {tool.avg_empty_result_rate:.1%} empty results. "
            f"Cross-referencing occurs in {cross.pct_claims_with_cross_references:.1f}% of claims, "
            f"and reasoning cites evidence in {cite.pct_reasoning_cites_evidence:.1f}% of cases "
            f"(avg citation quality: {cite.avg_reasoning_citation_score:.2f}/1.0)."
        )

    def _generate_utilization_patterns(self, metrics: SourceUtilizationMetrics) -> str:
        """Generate utilization pattern insights."""
        util = metrics.utilization
        fvs = metrics.failed_vs_successful
        corr = metrics.correlations

        insights = []

        # Overall utilization
        insights.append(
            f"Sources found vs marked useful: {util.avg_total_sources:.1f} total sources, "
            f"{util.avg_useful_sources:.1f} marked useful ({util.avg_utilization_rate:.1%} utilization rate)."
        )

        # Evolution over iterations
        if len(util.utilization_rate_by_iteration) > 1:
            first = util.utilization_rate_by_iteration[0]
            last = util.utilization_rate_by_iteration[-1]
            trend = "increases" if last > first else "decreases"
            insights.append(
                f"Utilization rate {trend} from {first:.1%} (iteration 1) to {last:.1%} (final iteration)."
            )

        # Failed vs successful
        diff = fvs.successful_avg_utilization_rate - fvs.failed_avg_utilization_rate
        if abs(diff) > 0.05:
            better = "Successful" if diff > 0 else "Failed"
            insights.append(
                f"{better} cases have higher utilization rates "
                f"({fvs.successful_avg_utilization_rate:.1%} vs {fvs.failed_avg_utilization_rate:.1%})."
            )

        # Correlation
        if corr.corr_utilization_rate_success:
            r, p = corr.corr_utilization_rate_success
            if p < 0.05:
                direction = "positively" if r > 0 else "negatively"
                insights.append(
                    f"Utilization rate is {direction} correlated with success (r={r:.3f}, p={p:.3f})."
                )

        return " ".join(insights)

    def _generate_tool_effectiveness_patterns(
        self, metrics: SourceUtilizationMetrics
    ) -> str:
        """Generate tool effectiveness pattern insights."""
        tool = metrics.tool_effectiveness
        fvs = metrics.failed_vs_successful
        corr = metrics.correlations

        insights = []

        # Overall effectiveness
        insights.append(
            f"Tool execution: {tool.total_successful_tools}/{tool.total_tool_calls} successful "
            f"({tool.avg_tool_success_rate:.1%} success rate). "
            f"Information loss: {tool.total_none_responses} NONE responses ({tool.avg_none_response_rate:.1%}), "
            f"{tool.total_empty_results} empty results ({tool.avg_empty_result_rate:.1%})."
        )

        # Failed vs successful
        success_diff = (
            fvs.successful_avg_tool_success_rate - fvs.failed_avg_tool_success_rate
        )
        none_diff = (
            fvs.failed_avg_none_response_rate - fvs.successful_avg_none_response_rate
        )

        if abs(success_diff) > 0.05:
            better = "Successful" if success_diff > 0 else "Failed"
            insights.append(
                f"{better} cases have higher tool success rates "
                f"({fvs.successful_avg_tool_success_rate:.1%} vs {fvs.failed_avg_tool_success_rate:.1%})."
            )

        if abs(none_diff) > 0.05:
            insights.append(
                f"Failed cases have more NONE responses "
                f"({fvs.failed_avg_none_response_rate:.1%} vs {fvs.successful_avg_none_response_rate:.1%}), "
                f"indicating information loss."
            )

        # Correlation
        if corr.corr_tool_success_rate_success:
            r, p = corr.corr_tool_success_rate_success
            if p < 0.05:
                direction = "positively" if r > 0 else "negatively"
                insights.append(
                    f"Tool success rate is {direction} correlated with success (r={r:.3f}, p={p:.3f})."
                )

        return " ".join(insights)

    def _generate_cross_referencing_patterns(
        self, metrics: SourceUtilizationMetrics
    ) -> str:
        """Generate cross-referencing pattern insights."""
        cross = metrics.cross_referencing
        fvs = metrics.failed_vs_successful
        corr = metrics.correlations

        insights = []

        # Overall cross-referencing
        insights.append(
            f"Cross-referencing: {cross.pct_claims_with_cross_references:.1f}% of claims have facts "
            f"corroborated by multiple sources. Average {cross.avg_num_cross_referenced_facts:.1f} facts "
            f"cross-referenced per claim, with {cross.avg_sources_per_fact:.1f} sources per fact."
        )

        # Failed vs successful
        diff = (
            fvs.successful_avg_cross_referenced_facts
            - fvs.failed_avg_cross_referenced_facts
        )
        if abs(diff) > 0.5:
            better = "Successful" if diff > 0 else "Failed"
            insights.append(
                f"{better} cases have more cross-referenced facts "
                f"({fvs.successful_avg_cross_referenced_facts:.1f} vs {fvs.failed_avg_cross_referenced_facts:.1f})."
            )

        # Correlation
        if corr.corr_cross_referencing_success:
            r, p = corr.corr_cross_referencing_success
            if p < 0.05:
                direction = "positively" if r > 0 else "negatively"
                insights.append(
                    f"Cross-referencing is {direction} correlated with success (r={r:.3f}, p={p:.3f})."
                )

        return " ".join(insights)

    def _generate_reasoning_citation_patterns(
        self, metrics: SourceUtilizationMetrics
    ) -> str:
        """Generate reasoning citation pattern insights."""
        cite = metrics.reasoning_citation
        fvs = metrics.failed_vs_successful
        corr = metrics.correlations

        insights = []

        # Overall citation
        insights.append(
            f"Evidence usage: Reasoning cites evidence in {cite.pct_reasoning_cites_evidence:.1f}% of cases, "
            f"with average citation quality of {cite.avg_reasoning_citation_score:.2f}/1.0."
        )

        # Failed vs successful
        diff = (
            fvs.successful_avg_reasoning_citation_score
            - fvs.failed_avg_reasoning_citation_score
        )
        if abs(diff) > 0.1:
            better = "Successful" if diff > 0 else "Failed"
            insights.append(
                f"{better} cases have better evidence integration "
                f"({fvs.successful_avg_reasoning_citation_score:.2f} vs {fvs.failed_avg_reasoning_citation_score:.2f})."
            )

        # Correlation
        if corr.corr_reasoning_citation_success:
            r, p = corr.corr_reasoning_citation_success
            if p < 0.05:
                direction = "positively" if r > 0 else "negatively"
                insights.append(
                    f"Reasoning citation quality is {direction} correlated with success (r={r:.3f}, p={p:.3f})."
                )

        return " ".join(insights)

    def _generate_success_factors(self, metrics: SourceUtilizationMetrics) -> str:
        """Generate insights on what distinguishes successful from failed cases."""
        fvs = metrics.failed_vs_successful

        factors = []

        # Identify key differences
        util_diff = (
            fvs.successful_avg_utilization_rate - fvs.failed_avg_utilization_rate
        )
        tool_diff = (
            fvs.successful_avg_tool_success_rate - fvs.failed_avg_tool_success_rate
        )
        cross_diff = (
            fvs.successful_avg_cross_referenced_facts
            - fvs.failed_avg_cross_referenced_facts
        )
        cite_diff = (
            fvs.successful_avg_reasoning_citation_score
            - fvs.failed_avg_reasoning_citation_score
        )

        if util_diff > 0.05:
            factors.append("higher source utilization rates")
        if tool_diff > 0.05:
            factors.append("better tool execution success")
        if cross_diff > 0.5:
            factors.append("more cross-referencing")
        if cite_diff > 0.1:
            factors.append("better evidence integration in reasoning")

        if factors:
            return f"Successful cases are characterized by: {', '.join(factors)}."
        else:
            return "No clear distinguishing factors between successful and failed cases based on source utilization metrics."

    def _generate_recommendations(self, metrics: SourceUtilizationMetrics) -> str:
        """Generate actionable recommendations."""
        util = metrics.utilization
        tool = metrics.tool_effectiveness
        cross = metrics.cross_referencing
        cite = metrics.reasoning_citation

        recommendations = []

        # Utilization recommendations
        if util.avg_utilization_rate < 0.5:
            recommendations.append(
                "Improve source utilization: Less than half of found sources are marked useful. "
                "Consider refining search queries or improving source filtering."
            )

        # Tool effectiveness recommendations
        if tool.avg_tool_success_rate < 0.8:
            recommendations.append(
                f"Address tool failures: {tool.avg_tool_success_rate:.1%} success rate indicates room for improvement. "
                "Investigate common failure patterns and improve error handling."
            )

        if tool.avg_none_response_rate > 0.2:
            recommendations.append(
                f"Reduce NONE responses: {tool.avg_none_response_rate:.1%} of queries return NONE. "
                "This suggests information loss. Improve query formulation or expand search strategies."
            )

        # Cross-referencing recommendations
        if cross.pct_claims_with_cross_references < 50:
            recommendations.append(
                "Increase cross-referencing: Less than half of claims have corroborating sources. "
                "Encourage searching for multiple perspectives and corroborating evidence."
            )

        # Citation recommendations
        if cite.avg_reasoning_citation_score < 0.6:
            recommendations.append(
                f"Improve evidence integration: Average citation quality is {cite.avg_reasoning_citation_score:.2f}/1.0. "
                "Ensure reasoning explicitly references and builds upon gathered evidence."
            )

        if not recommendations:
            recommendations.append(
                "Overall performance is good across all source utilization metrics."
            )

        return " ".join(recommendations)

    def _empty_metrics(self) -> SourceUtilizationMetrics:
        """Return empty metrics for cases with no data."""
        return SourceUtilizationMetrics(
            utilization=UtilizationMetrics(
                avg_utilization_rate=0.0,
                avg_total_sources=0.0,
                avg_useful_sources=0.0,
                avg_unique_sources=0.0,
                utilization_rate_by_iteration=[],
            ),
            tool_effectiveness=ToolEffectivenessMetrics(
                avg_tool_success_rate=0.0,
                avg_none_response_rate=0.0,
                avg_empty_result_rate=0.0,
                total_tool_calls=0,
                total_successful_tools=0,
                total_failed_tools=0,
                total_none_responses=0,
                total_empty_results=0,
            ),
            cross_referencing=CrossReferencingMetrics(
                avg_num_cross_referenced_facts=0.0,
                avg_sources_per_fact=0.0,
                pct_claims_with_cross_references=0.0,
            ),
            reasoning_citation=ReasoningCitationMetrics(
                avg_reasoning_citation_score=0.0, pct_reasoning_cites_evidence=0.0
            ),
            correlations=CorrelationMetrics(
                corr_utilization_rate_success=None,
                corr_tool_success_rate_success=None,
                corr_cross_referencing_success=None,
                corr_reasoning_citation_success=None,
            ),
            failed_vs_successful=FailedVsSuccessfulComparison(
                failed_avg_utilization_rate=0.0,
                successful_avg_utilization_rate=0.0,
                failed_avg_tool_success_rate=0.0,
                successful_avg_tool_success_rate=0.0,
                failed_avg_none_response_rate=0.0,
                successful_avg_none_response_rate=0.0,
                failed_avg_cross_referenced_facts=0.0,
                successful_avg_cross_referenced_facts=0.0,
                failed_avg_reasoning_citation_score=0.0,
                successful_avg_reasoning_citation_score=0.0,
            ),
        )
