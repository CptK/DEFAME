import multiprocessing
from tqdm import tqdm

from defame.analysis.analyzers.base_analyzer import BaseAnalyzer
from defame.analysis.analyzers.iteration.models import (
    IterationExtractedData,
    IterationMetrics,
    IterationInsights
)
from defame.analysis.analyzers.iteration.data_extraction import extract_iteration_data
from defame.analysis.data_models import ExperimentData, ClaimData
from defame.analysis.llm_helper import AnalyzerLLMHelper


def _process_claim_log_wrapper(args: tuple[dict, ClaimData, int]) -> IterationExtractedData | None:
    """Wrapper function for multiprocessing that recreates LLM helper from config."""
    config_dict, claim_data, max_iterations = args
    try:
        # Recreate LLM helper from config
        llm = AnalyzerLLMHelper.from_config_dict(config_dict)
        return extract_iteration_data(claim_data, llm, max_iterations)
    except Exception as e:
        print(f"Error processing claim {claim_data.claim_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


class IterationAnalyzer(BaseAnalyzer[IterationExtractedData, IterationMetrics, IterationInsights]):
    """Analyzer that extracts and aggregates iteration & stopping criteria data.

    Analyzes:
    - Number of iterations used vs available
    - Information gain per iteration
    - Stopping decision quality
    - Evidence accumulation patterns

    Example:
        from defame.analysis.analyzers.analyzer_config import AnalyzerConfig

        config = AnalyzerConfig(use_llm=True, use_multiprocessing=True)
        analyzer = IterationAnalyzer(config=config, max_iterations=5)
        result = analyzer.analyze(experiment_log)
    """

    def __init__(self, config, max_iterations: int = 5):
        """
        Initialize the analyzer.

        Args:
            config: Analyzer configuration
            max_iterations: Maximum iterations available to the system
        """
        super().__init__(config)
        self.max_iterations = max_iterations

    def extract_data(self, experiment_data: ExperimentData) -> list[IterationExtractedData]:
        if self.config.llm is None:
            raise ValueError("LLM configuration is required for IterationAnalyzer.")

        results = []
        if self.config.use_multiprocessing:
            # Serialize LLM config for multiprocessing
            config_dict = {
                'llm_model': self.config.llm.model.name,
                'llm_temperature': self.config.llm.model.temperature
            }

            # Create tuples of (config_dict, claim_data, max_iterations) for the wrapper function
            args_list = [
                (config_dict, claim_data, self.max_iterations)
                for claim_data in experiment_data.claims
            ]

            with multiprocessing.Pool() as pool:
                for result in pool.imap_unordered(_process_claim_log_wrapper, args_list):
                    if result is not None:
                        results.append(result)
        else:
            for claim_data in tqdm(experiment_data.claims, desc="Iteration Analysis", unit="claim"):
                result = extract_iteration_data(claim_data, self.config.llm, self.max_iterations)
                if result is not None:
                    results.append(result)
        return results

    def compute_metrics(self, extracted_data: list[IterationExtractedData]) -> IterationMetrics:
        if not extracted_data:
            raise ValueError("Cannot compute metrics on empty extracted data")

        # Separate failed and successful cases
        failed_claims = [claim for claim in extracted_data if not claim.prediction_correct]
        successful_claims = [claim for claim in extracted_data if claim.prediction_correct]

        # Iteration usage
        all_iterations = [claim.num_iterations for claim in extracted_data]
        avg_num_iterations = sum(all_iterations) / len(all_iterations)

        avg_num_iterations_wrong = (
            sum(claim.num_iterations for claim in failed_claims) / len(failed_claims)
            if failed_claims else 0.0
        )
        avg_num_iterations_correct = (
            sum(claim.num_iterations for claim in successful_claims) / len(successful_claims)
            if successful_claims else 0.0
        )

        using_max = sum(1 for claim in extracted_data if claim.num_iterations >= self.max_iterations)
        pct_using_max_iterations = 100.0 * using_max / len(extracted_data)

        # Information gain
        all_gains = []
        total_gains = []
        for claim in extracted_data:
            for iteration in claim.iterations:
                if iteration.information_gain > 0.0:
                    all_gains.append(iteration.information_gain)
            total_gains.append(claim.avg_information_gain * len(claim.iterations))

        avg_information_gain_per_iteration = sum(all_gains) / len(all_gains) if all_gains else 0.0
        avg_total_information_gain = sum(total_gains) / len(total_gains) if total_gains else 0.0

        # Information gain by iteration
        max_iter_count = max(claim.num_iterations for claim in extracted_data)
        information_gain_by_iteration = []
        for i in range(max_iter_count):
            gains = [
                claim.iterations[i].information_gain
                for claim in extracted_data
                if i < len(claim.iterations) and claim.iterations[i].information_gain > 0.0
            ]
            if gains:
                information_gain_by_iteration.append(sum(gains) / len(gains))

        # Evidence accumulation
        all_evidence = []
        total_evidence = []
        for claim in extracted_data:
            for iteration in claim.iterations:
                all_evidence.append(iteration.num_evidence_pieces)
            total_evidence.append(claim.total_evidence_pieces)

        avg_evidence_per_iteration = sum(all_evidence) / len(all_evidence) if all_evidence else 0.0
        avg_total_evidence_per_claim = sum(total_evidence) / len(total_evidence) if total_evidence else 0.0

        # Evidence by iteration
        evidence_by_iteration = []
        for i in range(max_iter_count):
            evidence_counts = [
                claim.iterations[i].num_evidence_pieces
                for claim in extracted_data
                if i < len(claim.iterations)
            ]
            if evidence_counts:
                evidence_by_iteration.append(sum(evidence_counts) / len(evidence_counts))

        # Stopping decisions
        stopping_decisions = [claim.stopping_decision for claim in extracted_data if claim.stopping_decision]
        total_decisions = len(stopping_decisions)

        appropriate = sum(1 for d in stopping_decisions if d.decision == "Appropriate")
        should_continue = sum(1 for d in stopping_decisions if d.decision == "Should_continue")
        uncertain = sum(1 for d in stopping_decisions if d.decision == "Uncertain")

        pct_appropriate_stopping = 100.0 * appropriate / total_decisions if total_decisions else 0.0
        pct_should_continue = 100.0 * should_continue / total_decisions if total_decisions else 0.0
        pct_uncertain_stopping = 100.0 * uncertain / total_decisions if total_decisions else 0.0

        # Early stopping in failures vs successes
        failures_stopped_early = sum(
            1 for claim in failed_claims
            if claim.num_iterations < self.max_iterations
        )
        successes_stopped_early = sum(
            1 for claim in successful_claims
            if claim.num_iterations < self.max_iterations
        )

        pct_failures_stopped_early = (
            100.0 * failures_stopped_early / len(failed_claims) if failed_claims else 0.0
        )
        pct_successes_stopped_early = (
            100.0 * successes_stopped_early / len(successful_claims) if successful_claims else 0.0
        )

        return IterationMetrics(
            avg_num_iterations=avg_num_iterations,
            avg_num_iterations_wrong=avg_num_iterations_wrong,
            avg_num_iterations_correct=avg_num_iterations_correct,
            pct_using_max_iterations=pct_using_max_iterations,
            avg_information_gain_per_iteration=avg_information_gain_per_iteration,
            avg_total_information_gain=avg_total_information_gain,
            information_gain_by_iteration=information_gain_by_iteration,
            avg_evidence_per_iteration=avg_evidence_per_iteration,
            avg_total_evidence_per_claim=avg_total_evidence_per_claim,
            evidence_by_iteration=evidence_by_iteration,
            pct_appropriate_stopping=pct_appropriate_stopping,
            pct_should_continue=pct_should_continue,
            pct_uncertain_stopping=pct_uncertain_stopping,
            pct_failures_stopped_early=pct_failures_stopped_early,
            pct_successes_stopped_early=pct_successes_stopped_early,
        )

    def generate_insights(self, metrics: IterationMetrics) -> IterationInsights:
        """
        Answers these questions:
        - Do failures use fewer iterations?
        - Do failures stop too early before finding key evidence?
        - What is the information gain pattern (plateau detection)?
        - Is stopping decision quality good?
        """
        # Question 1: Do failures use fewer iterations?
        iteration_gap = metrics.iteration_gap
        if abs(iteration_gap) > 0.2:
            if iteration_gap < 0:
                failures_iter_str = (
                    f"YES - Failures use fewer iterations on average. "
                    f"Failed predictions: {metrics.avg_num_iterations_wrong:.2f} iterations, "
                    f"Correct predictions: {metrics.avg_num_iterations_correct:.2f} iterations "
                    f"(gap: {abs(iteration_gap):.2f}). "
                    f"This suggests failures might benefit from more investigation."
                )
            else:
                failures_iter_str = (
                    f"NO - Failures actually use MORE iterations on average. "
                    f"Failed predictions: {metrics.avg_num_iterations_wrong:.2f} iterations, "
                    f"Correct predictions: {metrics.avg_num_iterations_correct:.2f} iterations "
                    f"(gap: {iteration_gap:.2f}). "
                    f"This suggests difficult claims take more iterations but still fail."
                )
        else:
            failures_iter_str = (
                f"NO SIGNIFICANT DIFFERENCE - Both use similar iterations. "
                f"Failed: {metrics.avg_num_iterations_wrong:.2f}, "
                f"Correct: {metrics.avg_num_iterations_correct:.2f} "
                f"(gap: {iteration_gap:.2f})."
            )

        # Question 2: Do failures stop too early?
        early_stop_gap = metrics.pct_failures_stopped_early - metrics.pct_successes_stopped_early
        if early_stop_gap > 10.0:
            early_stop_str = (
                f"YES - Failures stop early more often than successes. "
                f"{metrics.pct_failures_stopped_early:.1f}% of failures stopped before max iterations, "
                f"vs {metrics.pct_successes_stopped_early:.1f}% of successes. "
                f"Combined with {metrics.pct_should_continue:.1f}% flagged as 'should continue', "
                f"this suggests premature stopping contributes to failures."
            )
        elif early_stop_gap < -10.0:
            early_stop_str = (
                f"NO - Successes actually stop early more often. "
                f"{metrics.pct_successes_stopped_early:.1f}% of successes stopped before max iterations, "
                f"vs {metrics.pct_failures_stopped_early:.1f}% of failures. "
                f"This suggests easy claims are correctly identified and stopped early."
            )
        else:
            early_stop_str = (
                f"NO CLEAR PATTERN - Both stop early at similar rates. "
                f"Failures: {metrics.pct_failures_stopped_early:.1f}%, "
                f"Successes: {metrics.pct_successes_stopped_early:.1f}%."
            )

        # Question 3: Information gain pattern (plateau detection)
        if len(metrics.information_gain_by_iteration) >= 2:
            first_iter_gain = metrics.information_gain_by_iteration[0]
            last_iter_gain = metrics.information_gain_by_iteration[-1]
            gain_change = last_iter_gain - first_iter_gain

            if gain_change < -0.5:
                info_gain_str = (
                    f"DECLINING PATTERN - Information gain decreases across iterations. "
                    f"Iteration 0: {first_iter_gain:.2f}, Last iteration: {last_iter_gain:.2f} "
                    f"(change: {gain_change:.2f}). "
                    f"Per-iteration gains: {[f'{g:.2f}' for g in metrics.information_gain_by_iteration]}. "
                    f"This suggests a plateau effect where later iterations add diminishing value."
                )
            elif gain_change > 0.5:
                info_gain_str = (
                    f"INCREASING PATTERN - Information gain increases in later iterations. "
                    f"Iteration 0: {first_iter_gain:.2f}, Last iteration: {last_iter_gain:.2f} "
                    f"(change: {gain_change:.2f}). "
                    f"Per-iteration gains: {[f'{g:.2f}' for g in metrics.information_gain_by_iteration]}. "
                    f"This suggests persistence pays off with valuable later discoveries."
                )
            else:
                info_gain_str = (
                    f"STABLE PATTERN - Information gain remains relatively consistent. "
                    f"Average gain per iteration: {metrics.avg_information_gain_per_iteration:.2f}. "
                    f"Per-iteration gains: {[f'{g:.2f}' for g in metrics.information_gain_by_iteration]}. "
                    f"Each iteration contributes similarly to investigation."
                )
        else:
            info_gain_str = (
                f"INSUFFICIENT DATA - Most claims use only one iteration "
                f"(avg: {metrics.avg_num_iterations:.2f}). Cannot analyze gain patterns."
            )

        # Question 4: Stopping decision quality
        if metrics.pct_appropriate_stopping > 80.0 and metrics.pct_should_continue < 15.0:
            stopping_quality_str = (
                f"GOOD - Stopping decisions are mostly appropriate ({metrics.pct_appropriate_stopping:.1f}%). "
                f"Only {metrics.pct_should_continue:.1f}% should have continued, "
                f"{metrics.pct_uncertain_stopping:.1f}% uncertain. "
                f"The system generally knows when it has enough evidence."
            )
        elif metrics.pct_should_continue > 30.0 or metrics.pct_appropriate_stopping < 60.0:
            stopping_quality_str = (
                f"POOR - Too many claims stop prematurely ({metrics.pct_should_continue:.1f}% should have continued). "
                f"Only {metrics.pct_appropriate_stopping:.1f}% appropriate, "
                f"{metrics.pct_uncertain_stopping:.1f}% uncertain. "
                f"The stopping criteria are too aggressive - the system stops before gathering sufficient evidence."
            )
        else:
            stopping_quality_str = (
                f"NEEDS IMPROVEMENT - Stopping decisions are inconsistent. "
                f"{metrics.pct_appropriate_stopping:.1f}% appropriate, "
                f"{metrics.pct_should_continue:.1f}% should continue, "
                f"{metrics.pct_uncertain_stopping:.1f}% uncertain. "
                f"Stopping criteria should be refined to reduce premature stopping."
            )

        return IterationInsights(
            failures_use_fewer_iterations=failures_iter_str,
            early_stopping_in_failures=early_stop_str,
            information_gain_pattern=info_gain_str,
            stopping_decision_quality=stopping_quality_str,
        )
