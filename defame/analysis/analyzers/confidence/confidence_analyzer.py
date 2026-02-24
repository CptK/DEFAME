import multiprocessing
from scipy.stats import pearsonr
from tqdm import tqdm

from defame.analysis.analyzers.base_analyzer import BaseAnalyzer
from defame.analysis.analyzers.confidence.models import (
    ConfidenceExtractedData,
    ConfidenceMetrics,
    ConfidenceInsights
)
from defame.analysis.analyzers.confidence.data_extraction import extract_confidence_data
from defame.analysis.data_models import ClaimData, ExperimentData
from defame.analysis.llm_helper import AnalyzerLLMHelper


def _process_claim_log_wrapper(args: tuple[dict, ClaimData]) -> ConfidenceExtractedData | None:
    """Wrapper function for multiprocessing that recreates LLM helper from config."""
    config_dict, claim_data = args
    try:
        # Recreate LLM helper from config
        llm = AnalyzerLLMHelper.from_config_dict(config_dict)
        return extract_confidence_data(claim_data, llm)
    except Exception as e:
        print(f"Error processing claim log {claim_data.claim_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


class ConfidenceAnalyzer(BaseAnalyzer[ConfidenceExtractedData, ConfidenceMetrics, ConfidenceInsights]):
    """Analyzer that extracts and aggregates confidence data from claim logs.

    Extraction is done via LLM prompts for each iteration's judgement and evidence.

    Example:
        from defame.analysis.analyzers.analyzer_config import AnalyzerConfig

        config = AnalyzerConfig(use_llm=True, use_multiprocessing=True)
        analyzer = ConfidenceAnalyzer(config=config)
        result = analyzer.analyze(experiment_log)
    """

    def extract_data(self, experiment_data: ExperimentData) -> list[ConfidenceExtractedData]:
        if self.config.llm is None:
            raise ValueError("LLM configuration is required for ConfidenceAnalyzer.")

        results = []
        if self.config.use_multiprocessing:
            # Serialize LLM config for multiprocessing
            config_dict = {
                'llm_model': self.config.llm.model.name,
                'llm_temperature': self.config.llm.model.temperature
            }

            # Create tuples of (config_dict, claim_data) for the wrapper function
            args_list = [(config_dict, claim_data) for claim_data in experiment_data.claims]

            with multiprocessing.Pool() as pool:
                for result in pool.imap_unordered(_process_claim_log_wrapper, args_list):
                    if result is not None:
                        results.append(result)
        else:
            for claim_data in tqdm(experiment_data.claims, desc="Confidence Analysis", unit="claim"):
                result = extract_confidence_data(claim_data, self.config.llm)
                if result is not None:
                    results.append(result)
        return results
    
    def compute_metrics(self, extracted_data: list[ConfidenceExtractedData]) -> ConfidenceMetrics:
        total_confidence = 0.0
        total_evidence = 0.0
        total_hedging = 0.0
        count = 0

        confidence_list = []
        evidence_list = []
        hedging_list = []
        accuracy_list = []

        confidence_wrong = []
        evidence_wrong = []
        hedging_wrong = []
        confidence_correct = []
        evidence_correct = []
        hedging_correct = []

        for data in extracted_data:
            avg_conf = data.avg_confidence_strength
            avg_evid = data.avg_evidence_strength
            avg_hedge = data.avg_hedging_score

            total_confidence += avg_conf
            total_evidence += avg_evid
            total_hedging += avg_hedge
            count += 1

            confidence_list.append(avg_conf)
            evidence_list.append(avg_evid)
            hedging_list.append(avg_hedge)
            accuracy_list.append(1.0 if data.correct else 0.0)

            if data.correct:
                confidence_correct.append(avg_conf)
                evidence_correct.append(avg_evid)
                hedging_correct.append(avg_hedge)
            else:
                confidence_wrong.append(avg_conf)
                evidence_wrong.append(avg_evid)
                hedging_wrong.append(avg_hedge)

        avg_confidence_strength = total_confidence / count if count > 0 else 0.0
        avg_evidence_strength = total_evidence / count if count > 0 else 0.0
        avg_hedging_score = total_hedging / count if count > 0 else 0.0

        # Compute correlations safely
        def safe_correlation(list1: list[float], list2: list[float]) -> tuple[float, float] | None:
            """Compute correlation safely, returning None if invalid."""
            if len(list1) < 2 or len(set(list2)) < 2:
                return None
            try:
                r, p = pearsonr(list1, list2)
                assert isinstance(r, float) and isinstance(p, float)
                return (float(r), float(p))
            except Exception:
                return None

        corr_confidence_evidence = safe_correlation(confidence_list, evidence_list)
        corr_confidence_accuracy = safe_correlation(confidence_list, accuracy_list)
        corr_hedging_accuracy = safe_correlation(hedging_list, accuracy_list)

        avg_confidence_wrong = sum(confidence_wrong) / len(confidence_wrong) if confidence_wrong else 0.0
        avg_evidence_wrong = sum(evidence_wrong) / len(evidence_wrong) if evidence_wrong else 0.0
        avg_hedging_wrong = sum(hedging_wrong) / len(hedging_wrong) if hedging_wrong else 0.0
        avg_confidence_correct = sum(confidence_correct) / len(confidence_correct) if confidence_correct else 0.0
        avg_evidence_correct = sum(evidence_correct) / len(evidence_correct) if evidence_correct else 0.0
        avg_hedging_correct = sum(hedging_correct) / len(hedging_correct) if hedging_correct else 0.0

        # Compute within-claim confidence evolution
        # Track how confidence changes from first to last iteration for each claim
        confidence_changes = []
        claims_increased = 0
        claims_decreased = 0
        multi_iteration_claims = 0

        for data in extracted_data:
            # Only consider claims with multiple iterations and valid confidence scores
            valid_iterations = [it for it in data.iterations if it.confidence_strength > 0.0]
            if len(valid_iterations) >= 2:
                multi_iteration_claims += 1
                first_conf = valid_iterations[0].confidence_strength
                last_conf = valid_iterations[-1].confidence_strength
                change = last_conf - first_conf
                confidence_changes.append(change)

                if change > 0.0:
                    claims_increased += 1
                elif change < 0.0:
                    claims_decreased += 1

        avg_confidence_change_per_claim = (
            sum(confidence_changes) / len(confidence_changes) if confidence_changes else 0.0
        )
        pct_claims_confidence_increased = (
            100.0 * claims_increased / multi_iteration_claims if multi_iteration_claims > 0 else 0.0
        )
        pct_claims_confidence_decreased = (
            100.0 * claims_decreased / multi_iteration_claims if multi_iteration_claims > 0 else 0.0
        )

        return ConfidenceMetrics(
            avg_confidence_strength=avg_confidence_strength,
            avg_evidence_strength=avg_evidence_strength,
            corr_confidence_evidence=corr_confidence_evidence,
            corr_confidence_accuracy=corr_confidence_accuracy,
            avg_confidence_wrong=avg_confidence_wrong,
            avg_evidence_wrong=avg_evidence_wrong,
            avg_confidence_correct=avg_confidence_correct,
            avg_evidence_correct=avg_evidence_correct,
            avg_hedging_score=avg_hedging_score,
            avg_hedging_wrong=avg_hedging_wrong,
            avg_hedging_correct=avg_hedging_correct,
            corr_hedging_accuracy=corr_hedging_accuracy,
            avg_confidence_change_per_claim=avg_confidence_change_per_claim,
            pct_claims_confidence_increased=pct_claims_confidence_increased,
            pct_claims_confidence_decreased=pct_claims_confidence_decreased,
        )
    
    def generate_insights(self, metrics: ConfidenceMetrics) -> ConfidenceInsights:
        """
        Answers these questions:
        - Do failures show overconfidence?
        - Is there a confidence threshold that predicts failure?
        - Does hedging correlate with correctness when uncertain?

        Saves insights as human-readable strings.
        """
        # Question 1: Do failures show overconfidence?
        # Compare confidence between wrong and correct predictions
        confidence_gap = metrics.confidence_accuracy_gap
        if metrics.avg_confidence_wrong >= 3.5 and confidence_gap < 0.5:
            overconfidence_str = (
                f"YES - Failures show overconfidence. "
                f"Failed predictions have high confidence (avg={metrics.avg_confidence_wrong:.2f}) "
                f"similar to correct predictions (avg={metrics.avg_confidence_correct:.2f}, gap={confidence_gap:.2f}). "
                f"This suggests the model is overconfident even when wrong."
            )
        elif metrics.avg_confidence_wrong < metrics.avg_confidence_correct - 0.5:
            overconfidence_str = (
                f"NO - Failures show appropriate uncertainty. "
                f"Failed predictions have lower confidence (avg={metrics.avg_confidence_wrong:.2f}) "
                f"than correct predictions (avg={metrics.avg_confidence_correct:.2f}, gap={confidence_gap:.2f}). "
                f"The model is appropriately less confident when it fails."
            )
        else:
            overconfidence_str = (
                f"UNCLEAR - Confidence patterns are inconclusive. "
                f"Failed predictions: avg={metrics.avg_confidence_wrong:.2f}, "
                f"Correct predictions: avg={metrics.avg_confidence_correct:.2f}, gap={confidence_gap:.2f}."
            )

        # Question 2: Is there a confidence threshold that predicts failure?
        if metrics.corr_confidence_accuracy is not None:
            r, p = metrics.corr_confidence_accuracy
            if p < 0.05 and abs(r) > 0.2:
                threshold = metrics.avg_confidence_wrong
                threshold_str = (
                    f"YES - Confidence threshold identified at {threshold:.2f}. "
                    f"There is a significant correlation between confidence and accuracy (r={r:.3f}, p={p:.3f}). "
                    f"Predictions with confidence below {threshold:.2f} are more likely to fail."
                )
            else:
                threshold_str = (
                    f"NO - No clear confidence threshold found. "
                    f"Correlation between confidence and accuracy is not significant (r={r:.3f}, p={p:.3f}). "
                    f"Confidence alone does not reliably predict failure."
                )
        else:
            threshold_str = "INSUFFICIENT DATA - Cannot determine confidence threshold for failure."

        # Question 3: Does hedging correlate with correctness?
        hedging_gap = metrics.hedging_accuracy_gap
        if metrics.corr_hedging_accuracy is not None:
            r, p = metrics.corr_hedging_accuracy
            if p < 0.05 and abs(r) > 0.2:
                direction = "positive" if r > 0 else "negative"
                strength = "strong" if abs(r) > 0.5 else "moderate" if abs(r) > 0.3 else "weak"
                hedging_str = (
                    f"YES - {strength.upper()} {direction} correlation found (r={r:.3f}, p={p:.3f}). "
                    f"Failed predictions: avg hedging={metrics.avg_hedging_wrong:.2f}, "
                    f"Correct predictions: avg hedging={metrics.avg_hedging_correct:.2f}. "
                )
                if r > 0:
                    hedging_str += "Higher hedging is associated with correct predictions (model is uncertain even when correct)."
                else:
                    # Negative correlation: higher hedging → lower accuracy (more failures)
                    hedging_str += "Higher hedging is associated with failed predictions (model appropriately shows uncertainty when it fails)."
            else:
                hedging_str = (
                    f"NO - No significant correlation between hedging and correctness (r={r:.3f}, p={p:.3f}). "
                    f"Failed: avg hedging={metrics.avg_hedging_wrong:.2f}, "
                    f"Correct: avg hedging={metrics.avg_hedging_correct:.2f}."
                )
        else:
            hedging_str = "INSUFFICIENT DATA - Cannot determine hedging correlation with correctness."

        # Question 4: How does confidence evolve within individual claims?
        if metrics.avg_confidence_change_per_claim != 0.0:
            if metrics.avg_confidence_change_per_claim > 0.1:
                evolution_str = (
                    f"INCREASING - Confidence increases within claims across iterations (avg change: {metrics.avg_confidence_change_per_claim:+.2f}). "
                    f"{metrics.pct_claims_confidence_increased:.1f}% of multi-iteration claims show increasing confidence, "
                    f"{metrics.pct_claims_confidence_decreased:.1f}% show decreasing confidence. "
                    f"This suggests claims gain confidence as they gather more evidence over iterations."
                )
            elif metrics.avg_confidence_change_per_claim < -0.1:
                evolution_str = (
                    f"DECREASING - Confidence decreases within claims across iterations (avg change: {metrics.avg_confidence_change_per_claim:+.2f}). "
                    f"{metrics.pct_claims_confidence_decreased:.1f}% of multi-iteration claims show decreasing confidence, "
                    f"{metrics.pct_claims_confidence_increased:.1f}% show increasing confidence. "
                    f"This may indicate that additional evidence reveals complexity/uncertainty."
                )
            else:
                evolution_str = (
                    f"STABLE - Confidence remains relatively stable within claims (avg change: {metrics.avg_confidence_change_per_claim:+.2f}). "
                    f"{metrics.pct_claims_confidence_increased:.1f}% increase, {metrics.pct_claims_confidence_decreased:.1f}% decrease. "
                    f"Evidence accumulation doesn't strongly affect confidence."
                )
        else:
            evolution_str = "NO DATA - Not enough multi-iteration claims to analyze within-claim confidence evolution."

        return ConfidenceInsights(
            overconfidence_in_failures=overconfidence_str,
            confidence_threshold_for_failure=threshold_str,
            hedging_correlation_with_correctness=hedging_str,
            confidence_evolution_pattern=evolution_str,
        )
