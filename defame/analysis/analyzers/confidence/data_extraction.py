"""Data extraction logic for Confidence Analyzer."""

import re

from defame.analysis.analyzers.confidence.models import ConfidenceExtractedData, ConfidenceStrength
from defame.analysis.analyzers.confidence.prompts import (
    get_confidence_strength_prompt,
    get_evidence_strength_prompt,
    get_hedging_prompt,
)
from defame.analysis.data_models import ClaimData
from defame.analysis.llm_helper import AnalyzerLLMHelper
from defame.common.structured_log_schema import IterationInfo


def extract_confidence_data(
    claim_data: ClaimData, llm: AnalyzerLLMHelper | None = None
) -> ConfidenceExtractedData:
    """Extract confidence data for a single claim."""
    return ConfidenceExtractedData(
        claim_id=claim_data.claim_id,
        claim_text=claim_data.claim_text,
        prediction_correct=claim_data.correct,
        iterations=[
            _extract_iteration_confidence_data(it, claim_data.claim_text, i, llm) for i, it in enumerate(claim_data.log.iterations)
        ],
        correct=claim_data.correct,
    )

def _extract_iteration_confidence_data(
    iteration: IterationInfo,
    claim_text: str,
    iteration_num: int,
    llm: AnalyzerLLMHelper | None
) -> ConfidenceStrength:
    """Extract confidence data for a single iteration."""
    if llm is None:
        raise ValueError("LLM helper is required for confidence data extraction.")

    judgment = iteration.judgment
    confidence_strength = 0.0
    hedging_score = 0.0
    if judgment and judgment.reasoning:
        prompt = get_confidence_strength_prompt(judgment.reasoning.strip())
        response = llm.generate(prompt)
        if not isinstance(response, str):
            raise ValueError("LLM response for confidence strength is not a string.")
        confidence_strength = _extract_confidence_score_from_llm_response(response)

        # Also extract hedging language
        hedge_prompt = get_hedging_prompt(judgment.reasoning.strip())
        hedge_response = llm.generate(hedge_prompt)
        if not isinstance(hedge_response, str):
            raise ValueError("LLM response for hedging score is not a string.")
        hedging_score = _extract_confidence_score_from_llm_response(hedge_response)

    evidence_strength = 0.0
    # Extract evidence summaries from results
    evidence_summaries = []
    for action in iteration.evidence_retrieval:
        for result in action.results:
            if result.summary:
                evidence_summaries.append(result.summary)

    if evidence_summaries:
        prompt = get_evidence_strength_prompt(
            claim_text,
            evidence_summaries,
        )
        response = llm.generate(prompt)
        if not isinstance(response, str):
            raise ValueError("LLM response for evidence strength is not a string.")
        evidence_strength = _extract_confidence_score_from_llm_response(response)

    return ConfidenceStrength(
        iteration_num=iteration_num,
        confidence_strength=confidence_strength,
        evidence_strength=evidence_strength,
        hedging_score=hedging_score,
    )
    

def _extract_confidence_score_from_llm_response(response: str) -> float:
    """Extract confidence score (1-5) from LLM response.

    Handles multiple formats:
    - LaTeX boxed: $\boxed{4}$ or \boxed{4}
    - Plain numbers: 4 or 4.5
    - Labeled: "Score: 4", "Rating: 4"
    """
    # Try LaTeX boxed format first (most specific)
    match = re.search(r'\$?\\boxed\{(\d+(?:\.\d+)?)\}\$?', response)
    if match:
        score = float(match.group(1))
        if 1.0 <= score <= 5.0:
            return score

    # Try labeled format (Score: X, Rating: X, etc.)
    match = re.search(r'(?:score|rating|strength).*?[:\s]+(\d+(?:\.\d+)?)', response, re.IGNORECASE)
    if match:
        score = float(match.group(1))
        if 1.0 <= score <= 5.0:
            return score

    # Try any number at the end of the response
    match = re.search(r'(\d+(?:\.\d+)?)\s*$', response.strip())
    if match:
        score = float(match.group(1))
        if 1.0 <= score <= 5.0:
            return score

    # Try any number in the response (fallback)
    match = re.search(r'(\d+(?:\.\d+)?)', response)
    if match:
        score = float(match.group(1))
        if 1.0 <= score <= 5.0:
            return score

    raise ValueError(f"Could not extract valid confidence score from response: {response}")
