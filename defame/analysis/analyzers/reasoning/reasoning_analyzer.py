"""Reasoning Quality Analyzer."""

from __future__ import annotations

from collections import Counter
from multiprocessing import Pool
from typing import cast

import numpy as np
from scipy import stats
from tqdm import tqdm

from defame.analysis.analyzer_config import AnalyzerConfig
from defame.analysis.analyzers.base_analyzer import BaseAnalyzer
from defame.analysis.analyzers.reasoning.data_extraction import ReasoningExtractor
from defame.analysis.analyzers.reasoning.models import (
    CoherenceMetrics,
    EvidenceClaimMetrics,
    FailedVsSuccessfulReasoningComparison,
    FallacyMetrics,
    LogicalChainMetrics,
    ReasoningCorrelationMetrics,
    ReasoningExtractedData,
    ReasoningInsights,
    ReasoningMetrics,
    SynthesisMetrics,
)
from defame.analysis.llm_helper import AnalyzerLLMHelper
from defame.analysis.data_models import ClaimData, ExperimentData


def _process_claim_log_wrapper(
    args: tuple[dict, ClaimData]
) -> ReasoningExtractedData | None:
    """
    Wrapper function for multiprocessing to process a single claim.

    Args:
        args: Tuple of (config_dict, claim_data)

    Returns:
        ReasoningExtractedData or None if processing fails
    """
    config_dict, claim_data = args

    try:
        # Reconstruct LLM helper in worker process
        from defame.common.modeling import make_model

        llm_helper = None
        if config_dict.get("llm_config"):
            llm_config = config_dict["llm_config"]
            model_kwargs = {"temperature": llm_config["temperature"]}
            model = make_model(llm_config["model_name"], **model_kwargs)
            llm_helper = AnalyzerLLMHelper.from_model(model)

        extractor = ReasoningExtractor(llm_helper=llm_helper)
        return extractor.process_claim_log(claim_data)

    except Exception as e:
        print(f"Error processing claim {claim_data.claim_id}: {e}")
        import traceback

        traceback.print_exc()
        return None


class ReasoningAnalyzer(
    BaseAnalyzer[ReasoningExtractedData, ReasoningMetrics, ReasoningInsights]
):
    """Analyzer for reasoning quality in fact-checking."""

    def __init__(self, config: AnalyzerConfig) -> None:
        """
        Initialize the analyzer.

        Args:
            config: Analyzer configuration
        """
        super().__init__(config)

    def extract_data(self, experiment_data: ExperimentData) -> list[ReasoningExtractedData]:
        """
        Extract reasoning quality data from all claim logs.

        Args:
            experiment_log: Experiment log containing claim logs

        Returns:
            List of extracted data (one per claim)
        """
        # Serialize config to dict
        config_dict: dict = {"llm_config": None}

        if self.config.llm:
            config_dict["llm_config"] = {
                "model_name": self.config.llm.model.name,
                "temperature": self.config.llm.model.temperature,
            }

        if self.config.use_multiprocessing:
            # Use multiprocessing
            with Pool() as pool:
                args_list = [
                    (config_dict, claim_data) for claim_data in experiment_data.claims
                ]
                results = list(pool.imap_unordered(_process_claim_log_wrapper, args_list))
                extracted_data = [r for r in results if r is not None]
        else:
            # Single-threaded processing
            extractor = ReasoningExtractor(llm_helper=self.config.llm)
            extracted_data = []
            for claim_data in tqdm(experiment_data.claims, desc="Reasoning Analysis", unit="claim"):
                try:
                    data = extractor.process_claim_log(claim_data)
                    extracted_data.append(data)
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    print(f"Error processing claim {claim_data.claim_id}: {e}")

        return extracted_data

    def compute_metrics(
        self, extracted_data: list[ReasoningExtractedData]
    ) -> ReasoningMetrics:
        """
        Compute aggregated metrics from extracted data.

        Args:
            extracted_data: List of extracted data per claim

        Returns:
            Aggregated metrics
        """
        if not extracted_data:
            return self._empty_metrics()

        # Separate by success/failure
        failed_data = [d for d in extracted_data if not d.correct]
        successful_data = [d for d in extracted_data if d.correct]

        # === COHERENCE METRICS ===
        coherence_scores = [d.avg_logical_coherence for d in extracted_data]
        avg_coherence = float(np.mean(coherence_scores))

        # Compute per-iteration coherence
        max_iterations = max(
            len(d.iteration_reasoning)
            for d in extracted_data
            if d.iteration_reasoning
        )
        coherence_by_iteration = []
        for i in range(max_iterations):
            scores = [
                d.iteration_reasoning[i].logical_coherence_score
                for d in extracted_data
                if i < len(d.iteration_reasoning) and d.iteration_reasoning[i].has_reasoning
            ]
            if scores:
                coherence_by_iteration.append(float(np.mean(scores)))

        pct_high_coherence = (
            100.0
            * sum(1 for score in coherence_scores if score >= 4.0)
            / len(coherence_scores)
        )

        coherence = CoherenceMetrics(
            avg_coherence_score=avg_coherence,
            coherence_score_by_iteration=coherence_by_iteration,
            pct_high_coherence=pct_high_coherence,
        )

        # === EVIDENCE-CLAIM CONNECTION METRICS ===
        evidence_claim_scores = [d.avg_evidence_claim_strength for d in extracted_data]
        avg_evidence_claim_strength = float(np.mean(evidence_claim_scores))

        pct_addresses_claim = (
            100.0
            * sum(1 for d in extracted_data if d.pct_iterations_address_claim > 0.5)
            / len(extracted_data)
        )

        pct_strong_connection = (
            100.0
            * sum(1 for score in evidence_claim_scores if score >= 0.7)
            / len(evidence_claim_scores)
        )

        evidence_claim = EvidenceClaimMetrics(
            avg_evidence_claim_strength=avg_evidence_claim_strength,
            pct_addresses_claim=pct_addresses_claim,
            pct_strong_connection=pct_strong_connection,
        )

        # === LOGICAL CHAIN METRICS ===
        chain_scores = [d.avg_chain_strength for d in extracted_data]
        avg_chain_strength = float(np.mean(chain_scores))

        chain_breaks_per_claim = [d.total_chain_breaks for d in extracted_data]
        avg_chain_breaks = float(np.mean(chain_breaks_per_claim))

        pct_strong_chains = (
            100.0 * sum(1 for score in chain_scores if score >= 4.0) / len(chain_scores)
        )

        # Collect all chain breaks
        all_chain_breaks = []
        for d in extracted_data:
            for it in d.iteration_reasoning:
                all_chain_breaks.extend(it.chain_breaks)

        # Find common chain breaks (top 5)
        break_counter = Counter(all_chain_breaks)
        common_chain_breaks = break_counter.most_common(5)

        logical_chain = LogicalChainMetrics(
            avg_chain_strength=avg_chain_strength,
            avg_chain_breaks_per_claim=avg_chain_breaks,
            pct_strong_chains=pct_strong_chains,
            common_chain_breaks=common_chain_breaks,
        )

        # === SYNTHESIS METRICS ===
        synthesis_scores = [d.avg_synthesis_quality for d in extracted_data]
        avg_synthesis_quality = float(np.mean(synthesis_scores))

        # Compute per-iteration synthesis
        synthesis_by_iteration = []
        for i in range(max_iterations):
            scores = [
                d.iteration_reasoning[i].synthesis_quality_score
                for d in extracted_data
                if i < len(d.iteration_reasoning) and d.iteration_reasoning[i].has_reasoning
            ]
            if scores:
                synthesis_by_iteration.append(float(np.mean(scores)))

        pct_good_synthesis = (
            100.0
            * sum(1 for score in synthesis_scores if score >= 4.0)
            / len(synthesis_scores)
        )

        synthesis = SynthesisMetrics(
            avg_synthesis_quality=avg_synthesis_quality,
            synthesis_quality_by_iteration=synthesis_by_iteration,
            pct_good_synthesis=pct_good_synthesis,
        )

        # === FALLACY METRICS ===
        fallacy_counts = [d.total_logical_fallacies for d in extracted_data]
        avg_fallacies = float(np.mean(fallacy_counts))
        total_fallacies = sum(fallacy_counts)

        # Collect all fallacies
        all_fallacies = []
        for d in extracted_data:
            for it in d.iteration_reasoning:
                all_fallacies.extend([f.fallacy_type for f in it.logical_fallacies])

        # Find common fallacies (top 5)
        fallacy_counter = Counter(all_fallacies)
        common_fallacies = fallacy_counter.most_common(5)

        pct_with_fallacies = (
            100.0
            * sum(1 for count in fallacy_counts if count > 0)
            / len(fallacy_counts)
        )

        fallacies = FallacyMetrics(
            avg_fallacies_per_claim=avg_fallacies,
            total_fallacies=total_fallacies,
            common_fallacies=common_fallacies,
            pct_claims_with_fallacies=pct_with_fallacies,
        )

        # === CORRELATION METRICS ===
        success_labels = [1 if d.correct else 0 for d in extracted_data]

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

        correlations = ReasoningCorrelationMetrics(
            corr_coherence_success=safe_correlation(coherence_scores),
            corr_evidence_claim_strength_success=safe_correlation(evidence_claim_scores),
            corr_chain_strength_success=safe_correlation(chain_scores),
            corr_synthesis_quality_success=safe_correlation(synthesis_scores),
            corr_fallacy_count_success=safe_correlation(cast(list[float], fallacy_counts)),
        )

        # === FAILED VS SUCCESSFUL COMPARISON ===
        def safe_mean(data: list, attr: str) -> float:
            """Safely compute mean of an attribute."""
            if not data:
                return 0.0
            values = [getattr(d, attr) for d in data]
            return float(np.mean(values))

        failed_vs_successful = FailedVsSuccessfulReasoningComparison(
            failed_avg_coherence=safe_mean(failed_data, "avg_logical_coherence"),
            successful_avg_coherence=safe_mean(successful_data, "avg_logical_coherence"),
            failed_avg_evidence_claim_strength=safe_mean(
                failed_data, "avg_evidence_claim_strength"
            ),
            successful_avg_evidence_claim_strength=safe_mean(
                successful_data, "avg_evidence_claim_strength"
            ),
            failed_avg_chain_strength=safe_mean(failed_data, "avg_chain_strength"),
            successful_avg_chain_strength=safe_mean(successful_data, "avg_chain_strength"),
            failed_avg_synthesis_quality=safe_mean(failed_data, "avg_synthesis_quality"),
            successful_avg_synthesis_quality=safe_mean(
                successful_data, "avg_synthesis_quality"
            ),
            failed_avg_fallacies=safe_mean(failed_data, "total_logical_fallacies"),
            successful_avg_fallacies=safe_mean(successful_data, "total_logical_fallacies"),
        )

        return ReasoningMetrics(
            coherence=coherence,
            evidence_claim=evidence_claim,
            logical_chain=logical_chain,
            synthesis=synthesis,
            fallacies=fallacies,
            correlations=correlations,
            failed_vs_successful=failed_vs_successful,
        )

    def generate_insights(self, metrics: ReasoningMetrics) -> ReasoningInsights:
        """
        Generate human-readable insights from metrics.

        Args:
            metrics: Computed metrics

        Returns:
            Structured insights
        """
        summary = self._generate_summary(metrics)
        coherence_patterns = self._generate_coherence_patterns(metrics)
        evidence_integration_patterns = self._generate_evidence_integration_patterns(
            metrics
        )
        logical_chain_patterns = self._generate_logical_chain_patterns(metrics)
        synthesis_patterns = self._generate_synthesis_patterns(metrics)
        fallacy_patterns = self._generate_fallacy_patterns(metrics)
        failure_drivers = self._generate_failure_drivers(metrics)
        success_factors = self._generate_success_factors(metrics)
        recommendations = self._generate_recommendations(metrics)

        return ReasoningInsights(
            summary=summary,
            coherence_patterns=coherence_patterns,
            evidence_integration_patterns=evidence_integration_patterns,
            logical_chain_patterns=logical_chain_patterns,
            synthesis_patterns=synthesis_patterns,
            fallacy_patterns=fallacy_patterns,
            failure_drivers=failure_drivers,
            success_factors=success_factors,
            recommendations=recommendations,
        )

    def _generate_summary(self, metrics: ReasoningMetrics) -> str:
        """Generate high-level summary."""
        coh = metrics.coherence
        ev = metrics.evidence_claim
        chain = metrics.logical_chain
        synth = metrics.synthesis
        fall = metrics.fallacies

        return (
            f"Average reasoning coherence is {coh.avg_coherence_score:.2f}/5.0, with {coh.pct_high_coherence:.1f}% "
            f"achieving high coherence (≥4). Evidence-claim connection strength averages {ev.avg_evidence_claim_strength:.2f}/1.0, "
            f"with {ev.pct_strong_connection:.1f}% showing strong connections. Logical chain strength is {chain.avg_chain_strength:.2f}/5.0 "
            f"with {chain.avg_chain_breaks_per_claim:.1f} breaks per claim on average. Synthesis quality is {synth.avg_synthesis_quality:.2f}/5.0. "
            f"Logical fallacies appear in {fall.pct_claims_with_fallacies:.1f}% of claims ({fall.avg_fallacies_per_claim:.2f} per claim)."
        )

    def _generate_coherence_patterns(self, metrics: ReasoningMetrics) -> str:
        """Generate coherence pattern insights."""
        coh = metrics.coherence
        fvs = metrics.failed_vs_successful
        corr = metrics.correlations

        insights = []

        insights.append(
            f"Overall coherence: {coh.avg_coherence_score:.2f}/5.0 average, {coh.pct_high_coherence:.1f}% high coherence."
        )

        # Evolution over iterations
        if len(coh.coherence_score_by_iteration) > 1:
            first = coh.coherence_score_by_iteration[0]
            last = coh.coherence_score_by_iteration[-1]
            change = last - first
            if abs(change) > 0.3:
                trend = "improves" if change > 0 else "degrades"
                insights.append(
                    f"Coherence {trend} from {first:.2f} (iteration 1) to {last:.2f} (final iteration)."
                )

        # Failed vs successful
        diff = fvs.successful_avg_coherence - fvs.failed_avg_coherence
        if abs(diff) > 0.3:
            better = "Successful" if diff > 0 else "Failed"
            insights.append(
                f"{better} cases show higher coherence ({fvs.successful_avg_coherence:.2f} vs {fvs.failed_avg_coherence:.2f})."
            )

        # Correlation
        if corr.corr_coherence_success:
            r, p = corr.corr_coherence_success
            if p < 0.05:
                direction = "positively" if r > 0 else "negatively"
                insights.append(
                    f"Coherence is {direction} correlated with success (r={r:.3f}, p={p:.3f})."
                )

        return " ".join(insights)

    def _generate_evidence_integration_patterns(self, metrics: ReasoningMetrics) -> str:
        """Generate evidence integration pattern insights."""
        ev = metrics.evidence_claim
        fvs = metrics.failed_vs_successful
        corr = metrics.correlations

        insights = []

        insights.append(
            f"Evidence integration: {ev.avg_evidence_claim_strength:.2f}/1.0 average strength, "
            f"{ev.pct_addresses_claim:.1f}% address the claim, {ev.pct_strong_connection:.1f}% have strong connections."
        )

        # Failed vs successful
        diff = fvs.successful_avg_evidence_claim_strength - fvs.failed_avg_evidence_claim_strength
        if abs(diff) > 0.1:
            better = "Successful" if diff > 0 else "Failed"
            insights.append(
                f"{better} cases have stronger evidence-claim connections "
                f"({fvs.successful_avg_evidence_claim_strength:.2f} vs {fvs.failed_avg_evidence_claim_strength:.2f})."
            )

        # Correlation
        if corr.corr_evidence_claim_strength_success:
            r, p = corr.corr_evidence_claim_strength_success
            if p < 0.05:
                direction = "positively" if r > 0 else "negatively"
                insights.append(
                    f"Evidence-claim strength is {direction} correlated with success (r={r:.3f}, p={p:.3f})."
                )

        return " ".join(insights)

    def _generate_logical_chain_patterns(self, metrics: ReasoningMetrics) -> str:
        """Generate logical chain pattern insights."""
        chain = metrics.logical_chain
        fvs = metrics.failed_vs_successful
        corr = metrics.correlations

        insights = []

        insights.append(
            f"Logical chain: {chain.avg_chain_strength:.2f}/5.0 average strength, "
            f"{chain.avg_chain_breaks_per_claim:.1f} breaks per claim, {chain.pct_strong_chains:.1f}% strong chains."
        )

        # Common breaks
        if chain.common_chain_breaks:
            top_break = chain.common_chain_breaks[0]
            insights.append(f"Most common break: '{top_break[0]}' ({top_break[1]} occurrences).")

        # Failed vs successful
        diff = fvs.successful_avg_chain_strength - fvs.failed_avg_chain_strength
        if abs(diff) > 0.3:
            better = "Successful" if diff > 0 else "Failed"
            insights.append(
                f"{better} cases have stronger logical chains ({fvs.successful_avg_chain_strength:.2f} vs {fvs.failed_avg_chain_strength:.2f})."
            )

        # Correlation
        if corr.corr_chain_strength_success:
            r, p = corr.corr_chain_strength_success
            if p < 0.05:
                direction = "positively" if r > 0 else "negatively"
                insights.append(
                    f"Chain strength is {direction} correlated with success (r={r:.3f}, p={p:.3f})."
                )

        return " ".join(insights)

    def _generate_synthesis_patterns(self, metrics: ReasoningMetrics) -> str:
        """Generate synthesis pattern insights."""
        synth = metrics.synthesis
        fvs = metrics.failed_vs_successful
        corr = metrics.correlations

        insights = []

        insights.append(
            f"Synthesis quality: {synth.avg_synthesis_quality:.2f}/5.0 average, {synth.pct_good_synthesis:.1f}% good synthesis (≥4)."
        )

        # Evolution over iterations
        if len(synth.synthesis_quality_by_iteration) > 1:
            first = synth.synthesis_quality_by_iteration[0]
            last = synth.synthesis_quality_by_iteration[-1]
            change = last - first
            if abs(change) > 0.3:
                trend = "improves" if change > 0 else "degrades"
                insights.append(
                    f"Synthesis {trend} from {first:.2f} (iteration 1) to {last:.2f} (final iteration)."
                )

        # Failed vs successful
        diff = fvs.successful_avg_synthesis_quality - fvs.failed_avg_synthesis_quality
        if abs(diff) > 0.3:
            better = "Successful" if diff > 0 else "Failed"
            insights.append(
                f"{better} cases show better synthesis ({fvs.successful_avg_synthesis_quality:.2f} vs {fvs.failed_avg_synthesis_quality:.2f})."
            )

        # Correlation
        if corr.corr_synthesis_quality_success:
            r, p = corr.corr_synthesis_quality_success
            if p < 0.05:
                direction = "positively" if r > 0 else "negatively"
                insights.append(
                    f"Synthesis quality is {direction} correlated with success (r={r:.3f}, p={p:.3f})."
                )

        return " ".join(insights)

    def _generate_fallacy_patterns(self, metrics: ReasoningMetrics) -> str:
        """Generate fallacy pattern insights."""
        fall = metrics.fallacies
        fvs = metrics.failed_vs_successful

        insights = []

        insights.append(
            f"Fallacies: {fall.pct_claims_with_fallacies:.1f}% of claims contain fallacies "
            f"({fall.avg_fallacies_per_claim:.2f} per claim, {fall.total_fallacies} total)."
        )

        # Common fallacies
        if fall.common_fallacies:
            top_fallacy = fall.common_fallacies[0]
            insights.append(
                f"Most common fallacy: '{top_fallacy[0]}' ({top_fallacy[1]} occurrences)."
            )

        # Failed vs successful
        diff = fvs.failed_avg_fallacies - fvs.successful_avg_fallacies
        if abs(diff) > 0.3:
            more = "Failed" if diff > 0 else "Successful"
            insights.append(
                f"{more} cases have more fallacies ({fvs.failed_avg_fallacies:.2f} vs {fvs.successful_avg_fallacies:.2f})."
            )

        return " ".join(insights)

    def _generate_failure_drivers(self, metrics: ReasoningMetrics) -> str:
        """
        Answer: Is weak reasoning the primary driver of failure?
        """
        fvs = metrics.failed_vs_successful
        corr = metrics.correlations

        drivers = []

        # Check coherence difference
        coh_diff = fvs.successful_avg_coherence - fvs.failed_avg_coherence
        if coh_diff > 0.5:
            drivers.append(
                f"Poor logical coherence is a significant failure driver (failed: {fvs.failed_avg_coherence:.2f}, successful: {fvs.successful_avg_coherence:.2f})"
            )

        # Check evidence-claim connection
        ev_diff = fvs.successful_avg_evidence_claim_strength - fvs.failed_avg_evidence_claim_strength
        if ev_diff > 0.15:
            drivers.append(
                f"Weak evidence-claim connections contribute to failure (failed: {fvs.failed_avg_evidence_claim_strength:.2f}, successful: {fvs.successful_avg_evidence_claim_strength:.2f})"
            )

        # Check logical chain
        chain_diff = fvs.successful_avg_chain_strength - fvs.failed_avg_chain_strength
        if chain_diff > 0.5:
            drivers.append(
                f"Broken logical chains are associated with failure (failed: {fvs.failed_avg_chain_strength:.2f}, successful: {fvs.successful_avg_chain_strength:.2f})"
            )

        # Check synthesis
        synth_diff = fvs.successful_avg_synthesis_quality - fvs.failed_avg_synthesis_quality
        if synth_diff > 0.5:
            drivers.append(
                f"Poor synthesis across sources contributes to failure (failed: {fvs.failed_avg_synthesis_quality:.2f}, successful: {fvs.successful_avg_synthesis_quality:.2f})"
            )

        # Check fallacies
        fall_diff = fvs.failed_avg_fallacies - fvs.successful_avg_fallacies
        if fall_diff > 0.5:
            drivers.append(
                f"Logical fallacies are more common in failures (failed: {fvs.failed_avg_fallacies:.2f}, successful: {fvs.successful_avg_fallacies:.2f})"
            )

        # Check correlations for statistical significance
        sig_correlations = []
        if corr.corr_coherence_success and corr.corr_coherence_success[1] < 0.05:
            sig_correlations.append("coherence")
        if corr.corr_evidence_claim_strength_success and corr.corr_evidence_claim_strength_success[1] < 0.05:
            sig_correlations.append("evidence integration")
        if corr.corr_chain_strength_success and corr.corr_chain_strength_success[1] < 0.05:
            sig_correlations.append("logical chain strength")
        if corr.corr_synthesis_quality_success and corr.corr_synthesis_quality_success[1] < 0.05:
            sig_correlations.append("synthesis quality")

        if drivers:
            result = "YES, weak reasoning is a primary driver of failure. " + " ".join(drivers)
            if sig_correlations:
                result += f" Statistically significant correlations found for: {', '.join(sig_correlations)}."
            return result
        else:
            return "Reasoning quality shows minimal differences between failed and successful cases, suggesting other factors may be more important drivers of failure."

    def _generate_success_factors(self, metrics: ReasoningMetrics) -> str:
        """Answer: Do successful cases show better synthesis and reasoning?"""
        fvs = metrics.failed_vs_successful

        factors = []

        if fvs.successful_avg_coherence > fvs.failed_avg_coherence + 0.3:
            factors.append("higher logical coherence")
        if fvs.successful_avg_evidence_claim_strength > fvs.failed_avg_evidence_claim_strength + 0.1:
            factors.append("stronger evidence-claim connections")
        if fvs.successful_avg_chain_strength > fvs.failed_avg_chain_strength + 0.3:
            factors.append("more robust logical chains")
        if fvs.successful_avg_synthesis_quality > fvs.failed_avg_synthesis_quality + 0.3:
            factors.append("better synthesis across sources")
        if fvs.successful_avg_fallacies < fvs.failed_avg_fallacies - 0.3:
            factors.append("fewer logical fallacies")

        if factors:
            return f"YES, successful cases demonstrate better reasoning quality characterized by: {', '.join(factors)}."
        else:
            return "Reasoning quality is similar between successful and failed cases."

    def _generate_recommendations(self, metrics: ReasoningMetrics) -> str:
        """Generate actionable recommendations."""
        coh = metrics.coherence
        ev = metrics.evidence_claim
        chain = metrics.logical_chain
        synth = metrics.synthesis
        fall = metrics.fallacies

        recommendations = []

        # Coherence recommendations
        if coh.avg_coherence_score < 3.5:
            recommendations.append(
                f"Improve logical coherence: Average score is {coh.avg_coherence_score:.2f}/5.0. "
                "Ensure reasoning flows logically with clear connections between ideas."
            )

        # Evidence integration recommendations
        if ev.avg_evidence_claim_strength < 0.6:
            recommendations.append(
                f"Strengthen evidence-claim connections: Average strength is {ev.avg_evidence_claim_strength:.2f}/1.0. "
                "Ensure reasoning explicitly ties evidence to the specific claim being fact-checked."
            )

        # Logical chain recommendations
        if chain.avg_chain_strength < 3.5 or chain.avg_chain_breaks_per_claim > 1.0:
            recommendations.append(
                f"Improve logical chain flow: Average strength is {chain.avg_chain_strength:.2f}/5.0 with {chain.avg_chain_breaks_per_claim:.1f} breaks per claim. "
                "Ensure each inference is justified by the preceding evidence or reasoning."
            )

        # Synthesis recommendations
        if synth.avg_synthesis_quality < 3.5:
            recommendations.append(
                f"Enhance synthesis quality: Average score is {synth.avg_synthesis_quality:.2f}/5.0. "
                "Better integrate information across multiple sources, identify patterns, and address contradictions."
            )

        # Fallacy recommendations
        if fall.pct_claims_with_fallacies > 30:
            recommendations.append(
                f"Reduce logical fallacies: {fall.pct_claims_with_fallacies:.1f}% of claims contain fallacies. "
                "Implement fallacy detection and correction mechanisms in the reasoning process."
            )
            if fall.common_fallacies:
                top_fallacy = fall.common_fallacies[0][0]
                recommendations.append(f"Pay special attention to '{top_fallacy}' (most common fallacy).")

        if not recommendations:
            recommendations.append("Overall reasoning quality is strong across all metrics.")

        return " ".join(recommendations)

    def _empty_metrics(self) -> ReasoningMetrics:
        """Return empty metrics for cases with no data."""
        return ReasoningMetrics(
            coherence=CoherenceMetrics(
                avg_coherence_score=0.0,
                coherence_score_by_iteration=[],
                pct_high_coherence=0.0,
            ),
            evidence_claim=EvidenceClaimMetrics(
                avg_evidence_claim_strength=0.0,
                pct_addresses_claim=0.0,
                pct_strong_connection=0.0,
            ),
            logical_chain=LogicalChainMetrics(
                avg_chain_strength=0.0,
                avg_chain_breaks_per_claim=0.0,
                pct_strong_chains=0.0,
                common_chain_breaks=[],
            ),
            synthesis=SynthesisMetrics(
                avg_synthesis_quality=0.0,
                synthesis_quality_by_iteration=[],
                pct_good_synthesis=0.0,
            ),
            fallacies=FallacyMetrics(
                avg_fallacies_per_claim=0.0,
                total_fallacies=0,
                common_fallacies=[],
                pct_claims_with_fallacies=0.0,
            ),
            correlations=ReasoningCorrelationMetrics(
                corr_coherence_success=None,
                corr_evidence_claim_strength_success=None,
                corr_chain_strength_success=None,
                corr_synthesis_quality_success=None,
                corr_fallacy_count_success=None,
            ),
            failed_vs_successful=FailedVsSuccessfulReasoningComparison(
                failed_avg_coherence=0.0,
                successful_avg_coherence=0.0,
                failed_avg_evidence_claim_strength=0.0,
                successful_avg_evidence_claim_strength=0.0,
                failed_avg_chain_strength=0.0,
                successful_avg_chain_strength=0.0,
                failed_avg_synthesis_quality=0.0,
                successful_avg_synthesis_quality=0.0,
                failed_avg_fallacies=0.0,
                successful_avg_fallacies=0.0,
            ),
        )
