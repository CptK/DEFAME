"""Data extraction logic for Reasoning Quality Analyzer."""

import re
import warnings

from defame.analysis.analyzers.reasoning.models import (
    IterationReasoningQuality,
    LogicalFallacy,
    ReasoningExtractedData,
)
from defame.analysis.analyzers.reasoning.prompts import (
    get_evidence_claim_connection_prompt,
    get_logical_chain_prompt,
    get_logical_coherence_prompt,
    get_logical_fallacies_prompt,
    get_synthesis_quality_prompt,
)
from defame.analysis.llm_helper import AnalyzerLLMHelper
from defame.analysis.data_models import ClaimData


class ReasoningExtractor:
    """Extracts reasoning quality data from claim logs."""

    def __init__(self, llm_helper: AnalyzerLLMHelper | None = None):
        """
        Initialize the extractor.

        Args:
            llm_helper: LLM helper for reasoning analysis
        """
        self.llm_helper = llm_helper

    def extract_iteration_reasoning(
        self, iteration_num: int, claim_data: ClaimData
    ) -> IterationReasoningQuality:
        """
        Extract reasoning quality data for a single iteration.

        Args:
            iteration_num: Iteration number (0-indexed)
            claim_data: Claim log containing parsed iterations

        Returns:
            IterationReasoningQuality data
        """
        if iteration_num >= len(claim_data.log.iterations):
            return self._empty_iteration_reasoning(iteration_num)

        iteration = claim_data.log.iterations[iteration_num]

        # Check if elaboration exists
        if not iteration.elaboration or not iteration.elaboration.analysis_text:
            return self._empty_iteration_reasoning(iteration_num)

        reasoning_text = iteration.elaboration.analysis_text.strip()
        if not reasoning_text:
            return self._empty_iteration_reasoning(iteration_num)

        # Collect evidence from this iteration
        evidence = []
        for action in iteration.evidence_retrieval:
            for result in action.results:
                if result.marked_useful and result.summary:
                    evidence.append(result.summary)

        # If no LLM helper, return basic data
        if not self.llm_helper:
            return IterationReasoningQuality(
                iteration_num=iteration_num,
                has_reasoning=True,
                logical_coherence_score=0.0,
                coherence_explanation="LLM analysis not available",
                addresses_claim=False,
                evidence_claim_strength=0.0,
                connection_explanation="LLM analysis not available",
                logical_chain_strength=0.0,
                chain_breaks=[],
                chain_explanation="LLM analysis not available",
                logical_fallacies=[],
                synthesis_quality_score=0.0,
                synthesis_explanation="LLM analysis not available",
            )

        # Extract logical coherence
        coherence_score, coherence_explanation = self._extract_logical_coherence(
            claim_data.claim_text, reasoning_text, claim_data.claim_id
        )

        # Extract evidence-claim connection
        (
            addresses_claim,
            evidence_claim_strength,
            connection_explanation,
        ) = self._extract_evidence_claim_connection(
            claim_data.claim_text, evidence, reasoning_text, claim_data.claim_id
        )

        # Extract logical chain analysis
        chain_strength, chain_breaks, chain_explanation= self._extract_logical_chain(
            claim_data.claim_text, evidence, reasoning_text, claim_data.claim_id
        )

        # Determine if this is the final iteration
        is_final_iteration = (iteration_num == len(claim_data.log.iterations) - 1)

        # Extract logical fallacies with iteration context
        fallacies = self._extract_logical_fallacies(
            claim_data.claim_text,
            reasoning_text,
            claim_data.claim_id,
            iteration_num=iteration_num,
            is_final_iteration=is_final_iteration,
            prediction_correct=claim_data.correct if is_final_iteration else None
        )

        # Extract synthesis quality
        synthesis_score, synthesis_explanation = self._extract_synthesis_quality(
            claim_data.claim_text, evidence, reasoning_text, claim_data.claim_id
        )

        return IterationReasoningQuality(
            iteration_num=iteration_num,
            has_reasoning=True,
            logical_coherence_score=coherence_score,
            coherence_explanation=coherence_explanation,
            addresses_claim=addresses_claim,
            evidence_claim_strength=evidence_claim_strength,
            connection_explanation=connection_explanation,
            logical_chain_strength=chain_strength,
            chain_breaks=chain_breaks,
            chain_explanation=chain_explanation,
            logical_fallacies=fallacies,
            synthesis_quality_score=synthesis_score,
            synthesis_explanation=synthesis_explanation,
        )

    def _extract_logical_coherence(
        self, claim: str, reasoning: str, claim_id: str | None = None
    ) -> tuple[float, str]:
        """
        Extract logical coherence score and explanation.

        Args:
            claim: The claim being fact-checked
            reasoning: The reasoning text
            claim_id: Optional claim ID for error messages

        Returns:
            Tuple of (coherence_score: float, explanation: str)
        """
        if not self.llm_helper:
            return 0.0, "LLM not available"

        prompt = get_logical_coherence_prompt(claim, reasoning)

        try:
            response = self.llm_helper.generate(prompt, temperature=0.1)

            if not isinstance(response, str):
                claim_info = f" for claim {claim_id}" if claim_id else ""
                print(
                    f"Warning: Coherence extraction returned non-string{claim_info}: {type(response)}"
                )
                return 0.0, "LLM response error"

            # Parse COHERENCE_SCORE
            score_match = re.search(r"COHERENCE_SCORE:\s*([1-5])", response)
            score = float(score_match.group(1)) if score_match else 3.0

            # Parse EXPLANATION
            explanation_match = re.search(
                r"EXPLANATION:\s*(.+?)(?=\n|$)", response, re.DOTALL
            )
            explanation = (
                explanation_match.group(1).strip()
                if explanation_match
                else "No explanation provided"
            )

            return score, explanation

        except Exception as e:
            claim_info = f" for claim {claim_id}" if claim_id else ""
            print(f"Warning: Coherence extraction failed{claim_info}: {e}")
            return 0.0, f"Extraction error: {str(e)}"

    def _extract_evidence_claim_connection(
        self,
        claim: str,
        evidence: list[str],
        reasoning: str,
        claim_id: str | None = None,
    ) -> tuple[bool, float, str]:
        """
        Extract evidence-claim connection assessment.

        Args:
            claim: The claim being fact-checked
            evidence: List of evidence summaries
            reasoning: The reasoning text
            claim_id: Optional claim ID for error messages

        Returns:
            Tuple of (addresses_claim: bool, connection_strength: float, explanation: str)
        """
        if not self.llm_helper or not evidence:
            return False, 0.0, "No evidence or LLM not available"

        prompt = get_evidence_claim_connection_prompt(claim, evidence, reasoning)

        try:
            response = self.llm_helper.generate(prompt, temperature=0.1)

            if not isinstance(response, str):
                claim_info = f" for claim {claim_id}" if claim_id else ""
                print(
                    f"Warning: Evidence-claim extraction returned non-string{claim_info}: {type(response)}"
                )
                return False, 0.0, "LLM response error"

            # Parse ADDRESSES_CLAIM
            addresses_match = re.search(
                r"ADDRESSES_CLAIM:\s*([YN])", response, re.IGNORECASE
            )
            addresses_claim = (
                addresses_match.group(1).upper() == "Y" if addresses_match else False
            )

            # Parse CONNECTION_STRENGTH
            strength_match = re.search(r"CONNECTION_STRENGTH:\s*([\d.]+)", response)
            strength = float(strength_match.group(1)) if strength_match else 0.0
            strength = max(0.0, min(1.0, strength))  # Clamp to [0, 1]

            # Parse EXPLANATION
            explanation_match = re.search(
                r"EXPLANATION:\s*(.+?)(?=\n|$)", response, re.DOTALL
            )
            explanation = (
                explanation_match.group(1).strip()
                if explanation_match
                else "No explanation provided"
            )

            return addresses_claim, strength, explanation

        except Exception as e:
            claim_info = f" for claim {claim_id}" if claim_id else ""
            print(f"Warning: Evidence-claim extraction failed{claim_info}: {e}")
            return False, 0.0, f"Extraction error: {str(e)}"

    def _extract_logical_chain(
        self,
        claim: str,
        evidence: list[str],
        reasoning: str,
        claim_id: str | None = None,
    ) -> tuple[float, list[str], str]:
        """
        Extract logical chain analysis.

        Args:
            claim: The claim being fact-checked
            evidence: List of evidence summaries
            reasoning: The reasoning text
            claim_id: Optional claim ID for error messages

        Returns:
            Tuple of (chain_strength: float, chain_breaks: list[str], explanation: str)
        """
        if not self.llm_helper or not evidence:
            return 0.0, [], "No evidence or LLM not available"

        prompt = get_logical_chain_prompt(claim, evidence, reasoning)

        try:
            response = self.llm_helper.generate(prompt, temperature=0.1)

            if not isinstance(response, str):
                claim_info = f" for claim {claim_id}" if claim_id else ""
                print(
                    f"Warning: Logical chain extraction returned non-string{claim_info}: {type(response)}"
                )
                return 0.0, [], "LLM response error"

            # Parse CHAIN_STRENGTH
            strength_match = re.search(r"CHAIN_STRENGTH:\s*([1-5])", response)
            chain_strength = float(strength_match.group(1)) if strength_match else 3.0

            # Parse CHAIN_BREAKS
            breaks_match = re.search(
                r"CHAIN_BREAKS:\s*(.+?)(?=\nEXPLANATION:|$)", response, re.DOTALL
            )
            chain_breaks = []
            if breaks_match:
                breaks_text = breaks_match.group(1).strip()
                if breaks_text.lower() != "none" and breaks_text != "[]":
                    # Extract bullet points or lines
                    for line in breaks_text.split("\n"):
                        line = line.strip()
                        if line and line not in ["None", "none", "[]"]:
                            # Remove bullet points/dashes
                            line = re.sub(r"^[-•*]\s*", "", line)
                            if line:
                                chain_breaks.append(line)

            # Parse EXPLANATION
            explanation_match = re.search(
                r"EXPLANATION:\s*(.+?)(?=\n|$)", response, re.DOTALL
            )
            explanation = (
                explanation_match.group(1).strip()
                if explanation_match
                else "No explanation provided"
            )

            return chain_strength, chain_breaks, explanation

        except Exception as e:
            claim_info = f" for claim {claim_id}" if claim_id else ""
            print(f"Warning: Logical chain extraction failed{claim_info}: {e}")
            return 0.0, [], f"Extraction error: {str(e)}"

    def _extract_logical_fallacies(
        self,
        claim: str,
        reasoning: str,
        claim_id: str | None = None,
        iteration_num: int = 0,
        is_final_iteration: bool = False,
        prediction_correct: bool | None = None
    ) -> list[LogicalFallacy]:
        """
        Extract logical fallacies from reasoning.

        Args:
            claim: The claim being fact-checked
            reasoning: The reasoning text
            claim_id: Optional claim ID for error messages
            iteration_num: Current iteration number
            is_final_iteration: Whether this is the final iteration
            prediction_correct: Whether prediction was correct (for final iteration)

        Returns:
            List of LogicalFallacy objects
        """
        if not self.llm_helper:
            return []

        # check if llm_helper uses LLAMA model - it is highly unreliable for fallacy extraction
        if "llama" in self.llm_helper.model.name.lower():
            warnings.warn(
                "Using LLAMA model for logical fallacy extraction is not recommended due to reliability issues."
            )

        prompt = get_logical_fallacies_prompt(
            claim, reasoning, iteration_num, is_final_iteration, prediction_correct
        )

        try:
            result = self.llm_helper.extract_json(prompt, temperature=0.3)

            if not result:
                return []

            if not isinstance(result, list):
                claim_info = f" for claim {claim_id}" if claim_id else ""
                print(
                    f"Warning: Fallacy extraction returned non-list{claim_info}: {type(result)}"
                )
                return []

            fallacies = []
            for item in result:
                if not isinstance(item, dict):
                    continue

                fallacy_type = item.get("fallacy_type", "")
                description = item.get("description", "")

                if fallacy_type and description:
                    fallacies.append(
                        LogicalFallacy(
                            fallacy_type=fallacy_type, description=description
                        )
                    )

            return fallacies

        except Exception as e:
            claim_info = f" for claim {claim_id}" if claim_id else ""
            print(f"Warning: Fallacy extraction failed{claim_info}: {e}")
            return []

    def _extract_synthesis_quality(
        self,
        claim: str,
        evidence: list[str],
        reasoning: str,
        claim_id: str | None = None,
    ) -> tuple[float, str]:
        """
        Extract synthesis quality score.

        Args:
            claim: The claim being fact-checked
            evidence: List of evidence summaries
            reasoning: The reasoning text
            claim_id: Optional claim ID for error messages

        Returns:
            Tuple of (synthesis_score: float, explanation: str)
        """
        if not self.llm_helper or len(evidence) < 2:
            # Synthesis only applies when there are multiple sources
            return 0.0, "Less than 2 sources available"

        prompt = get_synthesis_quality_prompt(claim, evidence, reasoning)

        try:
            response = self.llm_helper.generate(prompt, temperature=0.1)

            if not isinstance(response, str):
                claim_info = f" for claim {claim_id}" if claim_id else ""
                print(
                    f"Warning: Synthesis extraction returned non-string{claim_info}: {type(response)}"
                )
                return 0.0, "LLM response error"

            # Parse SYNTHESIS_SCORE
            score_match = re.search(r"SYNTHESIS_SCORE:\s*([1-5])", response)
            score = float(score_match.group(1)) if score_match else 3.0

            # Parse EXPLANATION
            explanation_match = re.search(
                r"EXPLANATION:\s*(.+?)(?=\n|$)", response, re.DOTALL
            )
            explanation = (
                explanation_match.group(1).strip()
                if explanation_match
                else "No explanation provided"
            )

            return score, explanation

        except Exception as e:
            claim_info = f" for claim {claim_id}" if claim_id else ""
            print(f"Warning: Synthesis extraction failed{claim_info}: {e}")
            return 0.0, f"Extraction error: {str(e)}"

    def _empty_iteration_reasoning(self, iteration_num: int) -> IterationReasoningQuality:
        """Return empty reasoning quality for iterations without reasoning."""
        return IterationReasoningQuality(
            iteration_num=iteration_num,
            has_reasoning=False,
            logical_coherence_score=0.0,
            coherence_explanation="No reasoning in this iteration",
            addresses_claim=False,
            evidence_claim_strength=0.0,
            connection_explanation="No reasoning in this iteration",
            logical_chain_strength=0.0,
            chain_breaks=[],
            chain_explanation="No reasoning in this iteration",
            logical_fallacies=[],
            synthesis_quality_score=0.0,
            synthesis_explanation="No reasoning in this iteration",
        )

    def process_claim_log(self, claim_data: ClaimData) -> ReasoningExtractedData:
        """
        Process a single claim log and extract all reasoning quality data.

        Args:
            claim_data: Claim log to process

        Returns:
            ReasoningExtractedData object
        """
        # Extract per-iteration reasoning
        iteration_reasoning = []
        for i in range(len(claim_data.log.iterations)):
            iteration_reasoning.append(self.extract_iteration_reasoning(i, claim_data))

        return ReasoningExtractedData(
            claim_id=claim_data.claim_id,
            claim_text=claim_data.claim_text,
            prediction_correct=claim_data.correct,
            iteration_reasoning=iteration_reasoning,
            correct=claim_data.correct
        )
