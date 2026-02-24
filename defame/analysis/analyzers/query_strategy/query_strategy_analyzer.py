import multiprocessing
from collections import Counter
from typing import Any

import numpy as np
from scipy import stats
from tqdm import tqdm

from defame.analysis.data_models import ExperimentData, ClaimData
from defame.analysis.analyzers.base_analyzer import BaseAnalyzer
from defame.analysis.analyzer_config import AnalyzerConfig
from defame.analysis.analyzers.query_strategy.models import (
    QueryStrategyExtractedData,
    QueryStrategyMetrics,
    QueryStrategyInsights
)
from defame.analysis.analyzers.query_strategy.data_extraction import QueryStrategyDataExtractor


# Module-level function for multiprocessing (avoids pickling issues)
def _process_claim_log_wrapper(args: tuple[dict, ClaimData]) -> QueryStrategyExtractedData | None:
    """
    Wrapper function for multiprocessing that creates a fresh extractor per worker.

    Creates a new LLM helper instance in each worker to avoid pickling locks.
    Similar to DEFAME's worker.py pattern where models are initialized per-worker.
    """
    config_dict, claim_data = args

    # Reconstruct config with a fresh LLM helper in this worker process
    from defame.analysis.llm_helper import AnalyzerLLMHelper
    from defame.common.modeling import make_model

    llm_helper = None
    if config_dict.get('llm_config'):
        llm_config = config_dict['llm_config']
        # Create model with temperature parameter
        model_kwargs = {'temperature': llm_config['temperature']}
        model = make_model(llm_config['model_name'], **model_kwargs)
        llm_helper = AnalyzerLLMHelper.from_model(model)

    config = AnalyzerConfig(
        llm=llm_helper,
        use_multiprocessing=config_dict['use_multiprocessing'],
        metadata=config_dict.get('metadata', {})
    )

    extractor = QueryStrategyDataExtractor(config)
    return extractor.process_claim_data(claim_data)


class QueryStrategyAnalyzer(BaseAnalyzer[QueryStrategyExtractedData, QueryStrategyMetrics, QueryStrategyInsights]):
    """
    Analyzer that examines query strategies used during fact-checking.

    Examples:
        from defame.analysis.analyzers.analyzer_config import QueryStrategyAnalyzerConfig

        config = QueryStrategyAnalyzerConfig(
            llm_model_name="gpt4o_mini",
            llm_temperature=0.3,
            use_llm=True,
            use_multiprocessing=True
        )
        analyzer = QueryStrategyAnalyzer(config=config)

        # Analyze
        result = analyzer.analyze(experiment_log)
    """

    def __init__(self, config):
        super().__init__(config)
        self.data_extractor = QueryStrategyDataExtractor(config)

    def extract_data(self, experiment_data: ExperimentData) -> list[QueryStrategyExtractedData]:
        """
        Extract query strategy data from all claims in the experiment.

        For each claim, extracts:
        - Queries per iteration with specificity and overlap scores
        - Query evolution patterns
        - Tool diversity and redundancy
        - Tool choice coherence
        - Counter-evidence seeking behavior
        - Search angle analysis
        """
        results = []

        if self.config.use_multiprocessing:
            # Serialize config to dict (only serializable data, no LLM helper object)
            config_dict = {
                'use_multiprocessing': self.config.use_multiprocessing,
                'metadata': self.config.metadata,
                'llm_config': None
            }

            # If LLM is configured, save its initialization parameters
            if self.config.llm:
                config_dict['llm_config'] = {
                    'model_name': self.config.llm.model.name,
                    'temperature': self.config.llm.model.temperature
                }

            # Create tuples of (config_dict, claim_data) for the wrapper function
            args_list = [(config_dict, claim_data) for claim_data in experiment_data.claims]

            with multiprocessing.Pool() as pool:
                for result in pool.imap_unordered(_process_claim_log_wrapper, args_list):
                    if result is not None:
                        results.append(result)
        else:
            # Sequential processing (useful for debugging)
            for claim_data in tqdm(experiment_data.claims, desc="Query Strategy", unit="claim"):
                result = self.data_extractor.process_claim_data(claim_data)
                if result is not None:
                    results.append(result)

        return results

    def compute_metrics(self, extracted_data: list[QueryStrategyExtractedData]) -> QueryStrategyMetrics:
        """
        Compute aggregate metrics from all extracted claim data.

        This aggregates metrics across all claims:
        - Average queries per iteration
        - Average specificity/overlap trends
        - Correlations with success
        - Comparisons between failed and successful cases
        """
        if not extracted_data:
            raise ValueError("Cannot compute metrics on empty extracted data")

        # Separate failed and successful cases
        failed_claims = [claim for claim in extracted_data if not claim.success]
        successful_claims = [claim for claim in extracted_data if claim.success]

        # Determine max number of iterations to handle variable-length iteration lists
        max_iterations = max(len(claim.iterations) for claim in extracted_data)

        # Basic query statistics
        avg_num_queries_per_iteration = self._compute_avg_queries_per_iteration(extracted_data, max_iterations)
        avg_query_specificity = self._compute_avg_specificity(extracted_data)
        avg_query_specificity_per_iteration = self._compute_avg_specificity_per_iteration(extracted_data, max_iterations)
        avg_keyword_overlap = self._compute_avg_keyword_overlap(extracted_data)
        avg_keyword_overlap_per_iteration = self._compute_avg_keyword_overlap_per_iteration(extracted_data, max_iterations)

        # Search angle diversity
        avg_num_search_angles = float(np.mean([claim.search_angle_analysis.num_distinct_angles for claim in extracted_data]))

        # Query evolution metrics
        avg_specificity_change = self._compute_avg_specificity_change(extracted_data)
        avg_lexical_diversity = self._compute_avg_lexical_diversity(extracted_data)

        # Tool diversity metrics
        avg_tool_diversity = float(np.mean([claim.tool_diversity.diversity_score for claim in extracted_data]))
        avg_redundancy_ratio = float(np.mean([claim.tool_diversity.redundancy_ratio for claim in extracted_data]))
        most_common_tools = self._compute_most_common_tools(extracted_data)

        # Tool choice coherence
        avg_coherence = float(np.mean([claim.tool_choice_coherence.avg_coherence for claim in extracted_data]))

        # Counter-evidence seeking
        avg_counter_evidence_ratio = float(np.mean([claim.counter_evidence_seeking.counter_evidence_ratio for claim in extracted_data]))

        # Correlations with success
        corr_num_queries = self._compute_correlation(
            [sum(len(it.queries) for it in claim.iterations) for claim in extracted_data],
            [claim.success for claim in extracted_data]
        )
        corr_specificity = self._compute_correlation(
            [float(np.mean([it.avg_specificity for it in claim.iterations])) for claim in extracted_data],
            [claim.success for claim in extracted_data]
        )
        corr_tool_diversity = self._compute_correlation(
            [claim.tool_diversity.diversity_score for claim in extracted_data],
            [claim.success for claim in extracted_data]
        )
        corr_counter_evidence = self._compute_correlation(
            [claim.counter_evidence_seeking.counter_evidence_ratio for claim in extracted_data],
            [claim.success for claim in extracted_data]
        )

        # Comparison: failed vs successful
        comparison = self._compute_failed_vs_successful_comparison(failed_claims, successful_claims)

        return QueryStrategyMetrics(
            avg_num_queries_per_iteration=avg_num_queries_per_iteration,
            avg_query_specificity=avg_query_specificity,
            avg_query_specificity_per_iteration=avg_query_specificity_per_iteration,
            avg_keyword_overlap_with_claim=avg_keyword_overlap,
            avg_keyword_overlap_per_iteration=avg_keyword_overlap_per_iteration,
            avg_num_search_angles_used=avg_num_search_angles,
            avg_specificity_change_over_iterations=avg_specificity_change,
            avg_lexical_diversity_between_iterations=avg_lexical_diversity,
            avg_tool_diversity_score=avg_tool_diversity,
            avg_redundancy_ratio=avg_redundancy_ratio,
            most_common_tools=most_common_tools,
            avg_tool_choice_coherence=avg_coherence,
            avg_counter_evidence_ratio=avg_counter_evidence_ratio,
            corr_num_queries_success=corr_num_queries,
            corr_specificity_success=corr_specificity,
            corr_tool_diversity_success=corr_tool_diversity,
            corr_counter_evidence_success=corr_counter_evidence,
            failed_vs_successful_comparison=comparison
        )

    def generate_insights(self, metrics: QueryStrategyMetrics) -> QueryStrategyInsights:
        """
        Generate human-readable insights from metrics.

        Answers key questions:
        - Are failed cases more vague in queries?
        - Do they repeat query phrasing?
        - Do queries get more/less specific over iterations?
        - Tool diversity patterns
        - Counter-evidence seeking behavior
        """
        # Generate individual insights
        query_vagueness = self._generate_query_vagueness_insight(metrics)
        query_repetition = self._generate_query_repetition_insight(metrics)
        specificity_evolution = self._generate_specificity_evolution_insight(metrics)
        tool_diversity = self._generate_tool_diversity_insight(metrics)
        counter_evidence = self._generate_counter_evidence_insight(metrics)
        correlations = self._generate_correlation_insight(metrics)

        # Generate overall summary
        summary = self._generate_overall_summary(metrics)

        return QueryStrategyInsights(
            insights_summary=summary,
            query_vagueness_in_failures=query_vagueness,
            query_repetition_patterns=query_repetition,
            specificity_evolution=specificity_evolution,
            tool_diversity_insights=tool_diversity,
            counter_evidence_insights=counter_evidence,
            correlation_insights=correlations
        )

    # ========================================================================
    # Helper methods for generate_insights
    # ========================================================================

    def _generate_query_vagueness_insight(self, metrics: QueryStrategyMetrics) -> str:
        """Generate insight about query vagueness in failures."""
        comp = metrics.failed_vs_successful_comparison

        if not comp:
            return "Insufficient data to compare failed and successful cases."

        failed_spec = comp.get("avg_specificity_failed", 0)
        success_spec = comp.get("avg_specificity_successful", 0)
        diff = success_spec - failed_spec
        diff_pct = (diff / failed_spec * 100) if failed_spec > 0 else 0

        if abs(diff) < 0.1:
            return f"Failed and successful cases show similar query specificity (both ~{failed_spec:.2f}/5). Query vagueness does not appear to distinguish success from failure."
        elif diff > 0:
            return f"Successful cases use MORE specific queries (avg {success_spec:.2f}/5) compared to failed cases ({failed_spec:.2f}/5), a difference of {diff_pct:.1f}%. This suggests that vague queries may contribute to failures."
        else:
            return f"Failed cases surprisingly use MORE specific queries (avg {failed_spec:.2f}/5) compared to successful cases ({success_spec:.2f}/5). Query specificity alone does not explain failures."

    def _generate_query_repetition_insight(self, metrics: QueryStrategyMetrics) -> str:
        """Generate insight about query repetition patterns."""
        comp = metrics.failed_vs_successful_comparison

        if not comp:
            return "Insufficient data to compare failed and successful cases."

        failed_redundancy = comp.get("avg_redundancy_failed", 0)
        success_redundancy = comp.get("avg_redundancy_successful", 0)
        overall_redundancy = metrics.avg_redundancy_ratio

        diff = failed_redundancy - success_redundancy
        diff_pct = (diff / success_redundancy * 100) if success_redundancy > 0 else 0

        insight = f"Overall, {overall_redundancy * 100:.1f}% of queries are redundant (similar to previous queries). "

        if abs(diff) < 0.05:
            insight += f"Both failed ({failed_redundancy * 100:.1f}%) and successful ({success_redundancy * 100:.1f}%) cases show similar redundancy rates."
        elif diff > 0:
            insight += f"Failed cases show HIGHER redundancy ({failed_redundancy * 100:.1f}%) compared to successful cases ({success_redundancy * 100:.1f}%), suggesting failed cases may repeat query phrasing more often."
        else:
            insight += f"Successful cases show slightly higher redundancy ({success_redundancy * 100:.1f}%) compared to failed cases ({failed_redundancy * 100:.1f}%)."

        return insight

    def _generate_specificity_evolution_insight(self, metrics: QueryStrategyMetrics) -> str:
        """Generate insight about how specificity changes over iterations."""
        spec_trend = metrics.avg_query_specificity_per_iteration
        spec_change = metrics.avg_specificity_change_over_iterations

        if len(spec_trend) < 2:
            return "Not enough iterations to analyze specificity evolution."

        first_iter = spec_trend[0]
        last_iter = spec_trend[-1]

        insight = f"Query specificity across iterations: " + " → ".join([f"{s:.2f}" for s in spec_trend]) + ". "

        if abs(spec_change) < 0.1:
            insight += "Specificity remains relatively stable across iterations."
        elif spec_change > 0:
            insight += f"Queries become MORE specific over iterations (avg change: +{spec_change:.2f}), suggesting agents refine their search strategies."
        else:
            insight += f"Queries become LESS specific over iterations (avg change: {spec_change:.2f}), which may indicate broadening search or exhaustion of specific leads."

        return insight

    def _generate_tool_diversity_insight(self, metrics: QueryStrategyMetrics) -> str:
        """Generate insight about tool usage diversity."""
        comp = metrics.failed_vs_successful_comparison

        if not comp:
            return "Insufficient data to compare failed and successful cases."

        failed_diversity = comp.get("avg_tool_diversity_failed", 0)
        success_diversity = comp.get("avg_tool_diversity_successful", 0)
        diff = success_diversity - failed_diversity

        top_tools = metrics.most_common_tools[:3]
        tool_str = ", ".join([f"{tool} ({count})" for tool, count in top_tools])

        insight = f"Most used tools: {tool_str}. "
        insight += f"Average tool diversity score: {metrics.avg_tool_diversity_score:.3f}. "

        if abs(diff) < 0.01:
            insight += f"Failed and successful cases show similar tool diversity."
        elif diff > 0:
            insight += f"Successful cases show HIGHER tool diversity ({success_diversity:.3f}) vs failed cases ({failed_diversity:.3f}), suggesting varied approaches may improve success."
        else:
            insight += f"Failed cases show higher tool diversity ({failed_diversity:.3f}) vs successful cases ({success_diversity:.3f})."

        return insight

    def _generate_counter_evidence_insight(self, metrics: QueryStrategyMetrics) -> str:
        """Generate insight about counter-evidence seeking."""
        comp = metrics.failed_vs_successful_comparison

        if not comp:
            return "Insufficient data to compare failed and successful cases."

        overall_ratio = metrics.avg_counter_evidence_ratio
        failed_ratio = comp.get("avg_counter_evidence_ratio_failed", 0)
        success_ratio = comp.get("avg_counter_evidence_ratio_successful", 0)

        insight = f"Overall, {overall_ratio * 100:.1f}% of queries seek counter-evidence. "

        diff = success_ratio - failed_ratio
        if abs(diff) < 0.02:
            insight += f"Both failed ({failed_ratio * 100:.1f}%) and successful ({success_ratio * 100:.1f}%) cases seek counter-evidence at similar rates."
        elif diff > 0:
            insight += f"Successful cases seek counter-evidence MORE often ({success_ratio * 100:.1f}%) compared to failed cases ({failed_ratio * 100:.1f}%), suggesting critical evaluation may improve outcomes."
        else:
            insight += f"Failed cases seek counter-evidence more often ({failed_ratio * 100:.1f}%) compared to successful cases ({success_ratio * 100:.1f}%)."

        return insight

    def _generate_correlation_insight(self, metrics: QueryStrategyMetrics) -> str:
        """Generate insight about correlations with success."""
        insights = []

        # Number of queries correlation
        corr, p_val = metrics.corr_num_queries_success
        if p_val < 0.05:
            direction = "positively" if corr > 0 else "negatively"
            insights.append(f"Number of queries is {direction} correlated with success (r={corr:.3f}, p={p_val:.3f})")

        # Specificity correlation
        corr, p_val = metrics.corr_specificity_success
        if p_val < 0.05:
            direction = "positively" if corr > 0 else "negatively"
            insights.append(f"Query specificity is {direction} correlated with success (r={corr:.3f}, p={p_val:.3f})")

        # Tool diversity correlation
        corr, p_val = metrics.corr_tool_diversity_success
        if p_val < 0.05:
            direction = "positively" if corr > 0 else "negatively"
            insights.append(f"Tool diversity is {direction} correlated with success (r={corr:.3f}, p={p_val:.3f})")

        # Counter-evidence correlation
        corr, p_val = metrics.corr_counter_evidence_success
        if p_val < 0.05:
            direction = "positively" if corr > 0 else "negatively"
            insights.append(f"Counter-evidence seeking is {direction} correlated with success (r={corr:.3f}, p={p_val:.3f})")

        if not insights:
            return "No statistically significant correlations found between query strategies and success (all p > 0.05)."

        return "Significant correlations with success: " + "; ".join(insights) + "."

    def _generate_overall_summary(self, metrics: QueryStrategyMetrics) -> str:
        """Generate an overall summary of findings."""
        summary_parts = []

        # Query volume
        avg_queries = sum(metrics.avg_num_queries_per_iteration)
        summary_parts.append(f"Agents make an average of {avg_queries:.1f} queries across all iterations")

        # Specificity
        if metrics.avg_query_specificity > 0:
            summary_parts.append(f"with average specificity of {metrics.avg_query_specificity:.2f}/5")

        # Key finding from comparison
        comp = metrics.failed_vs_successful_comparison
        if comp:
            failed_spec = comp.get("avg_specificity_failed", 0)
            success_spec = comp.get("avg_specificity_successful", 0)
            if abs(success_spec - failed_spec) > 0.2:
                if success_spec > failed_spec:
                    summary_parts.append("Notably, successful cases use more specific queries than failed cases")
                else:
                    summary_parts.append("Interestingly, failed cases use more specific queries than successful ones")

        summary = ". ".join(summary_parts) + "."
        return summary

    # ========================================================================
    # Helper methods for compute_metrics
    # ========================================================================

    def _compute_avg_queries_per_iteration(
        self,
        data: list[QueryStrategyExtractedData],
        max_iterations: int
    ) -> list[float]:
        """Compute average number of queries for each iteration position."""
        queries_per_iteration = [[] for _ in range(max_iterations)]

        for claim in data:
            for i, iteration in enumerate(claim.iterations):
                queries_per_iteration[i].append(iteration.num_queries)

        return [float(np.mean(queries)) if queries else 0.0 for queries in queries_per_iteration]

    def _compute_avg_specificity(self, data: list[QueryStrategyExtractedData]) -> float:
        """Compute average specificity across all queries in all claims."""
        all_specificities = []
        for claim in data:
            for iteration in claim.iterations:
                for query in iteration.queries:
                    if query.specificity_score is not None:
                        all_specificities.append(query.specificity_score)

        return float(np.mean(all_specificities)) if all_specificities else 0.0

    def _compute_avg_specificity_per_iteration(
        self,
        data: list[QueryStrategyExtractedData],
        max_iterations: int
    ) -> list[float]:
        """Compute average specificity for each iteration position."""
        specificities_per_iteration = [[] for _ in range(max_iterations)]

        for claim in data:
            for i, iteration in enumerate(claim.iterations):
                specificities_per_iteration[i].append(iteration.avg_specificity)

        return [float(np.mean(specs)) if specs else 0.0 for specs in specificities_per_iteration]

    def _compute_avg_keyword_overlap(self, data: list[QueryStrategyExtractedData]) -> float:
        """Compute average keyword overlap across all queries in all claims."""
        all_overlaps = []
        for claim in data:
            for iteration in claim.iterations:
                for query in iteration.queries:
                    if query.keyword_overlap_with_claim is not None:
                        all_overlaps.append(query.keyword_overlap_with_claim)

        return float(np.mean(all_overlaps)) if all_overlaps else 0.0

    def _compute_avg_keyword_overlap_per_iteration(
        self,
        data: list[QueryStrategyExtractedData],
        max_iterations: int
    ) -> list[float]:
        """Compute average keyword overlap for each iteration position."""
        overlaps_per_iteration = [[] for _ in range(max_iterations)]

        for claim in data:
            for i, iteration in enumerate(claim.iterations):
                overlaps_per_iteration[i].append(iteration.avg_keyword_overlap)

        return [float(np.mean(overlaps)) if overlaps else 0.0 for overlaps in overlaps_per_iteration]

    def _compute_avg_specificity_change(self, data: list[QueryStrategyExtractedData]) -> float:
        """Compute average change in specificity from first to last iteration."""
        changes = []
        for claim in data:
            if len(claim.iterations) >= 2:
                first_spec = claim.iterations[0].avg_specificity
                last_spec = claim.iterations[-1].avg_specificity
                if first_spec > 0:  # Avoid division issues
                    changes.append(last_spec - first_spec)

        return float(np.mean(changes)) if changes else 0.0

    def _compute_avg_lexical_diversity(self, data: list[QueryStrategyExtractedData]) -> float:
        """Compute average lexical diversity across all claims."""
        all_diversities = []
        for claim in data:
            all_diversities.extend(claim.query_evolution.lexical_diversity_between_iterations)

        return float(np.mean(all_diversities)) if all_diversities else 0.0

    def _compute_most_common_tools(self, data: list[QueryStrategyExtractedData]) -> list[tuple[str, int]]:
        """Get the most commonly used tools across all claims."""
        all_tool_counts = Counter()
        for claim in data:
            all_tool_counts.update(claim.tool_diversity.tool_type_counts)

        return all_tool_counts.most_common(10)  # Return top 10

    def _compute_correlation(self, values: list[float], success: list[bool]) -> tuple[float, float]:
        """Compute Pearson correlation between a metric and success."""
        try:
            # Convert success to numeric (True=1, False=0)
            success_numeric = [1 if s else 0 for s in success]

            # Remove any NaN or inf values
            valid_pairs = [(v, s) for v, s in zip(values, success_numeric)
                          if not np.isnan(v) and not np.isinf(v)]

            if len(valid_pairs) < 2:
                return (0.0, 1.0)  # No correlation if not enough data

            vals, succs = zip(*valid_pairs)
            correlation, p_value = stats.pearsonr(vals, succs)
            assert isinstance(correlation, np.ndarray) and isinstance(p_value, np.ndarray)

            return (float(correlation), float(p_value))
        except Exception:
            return (0.0, 1.0)

    def _compute_failed_vs_successful_comparison(
        self,
        failed_claims: list[QueryStrategyExtractedData],
        successful_claims: list[QueryStrategyExtractedData]
    ) -> dict[str, Any]:
        """Compare metrics between failed and successful cases."""
        comparison = {}

        if failed_claims and successful_claims:
            # Average number of queries
            comparison["avg_num_queries_failed"] = float(np.mean([
                sum(len(it.queries) for it in claim.iterations) for claim in failed_claims
            ]))
            comparison["avg_num_queries_successful"] = float(np.mean([
                sum(len(it.queries) for it in claim.iterations) for claim in successful_claims
            ]))

            # Average specificity
            comparison["avg_specificity_failed"] = float(np.mean([
                float(np.mean([it.avg_specificity for it in claim.iterations])) for claim in failed_claims
            ]))
            comparison["avg_specificity_successful"] = float(np.mean([
                float(np.mean([it.avg_specificity for it in claim.iterations])) for claim in successful_claims
            ]))

            # Average redundancy
            comparison["avg_redundancy_failed"] = float(np.mean([
                claim.tool_diversity.redundancy_ratio for claim in failed_claims
            ]))
            comparison["avg_redundancy_successful"] = float(np.mean([
                claim.tool_diversity.redundancy_ratio for claim in successful_claims
            ]))

            # Average tool diversity
            comparison["avg_tool_diversity_failed"] = float(np.mean([
                claim.tool_diversity.diversity_score for claim in failed_claims
            ]))
            comparison["avg_tool_diversity_successful"] = float(np.mean([
                claim.tool_diversity.diversity_score for claim in successful_claims
            ]))

            # Counter-evidence seeking
            comparison["avg_counter_evidence_ratio_failed"] = float(np.mean([
                claim.counter_evidence_seeking.counter_evidence_ratio for claim in failed_claims
            ]))
            comparison["avg_counter_evidence_ratio_successful"] = float(np.mean([
                claim.counter_evidence_seeking.counter_evidence_ratio for claim in successful_claims
            ]))

        return comparison
