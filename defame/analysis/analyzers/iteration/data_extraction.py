"""Data extraction logic for Iteration & Stopping Criteria Analyzer."""

import re

from defame.analysis.analyzers.iteration.models import (
    IterationExtractedData,
    IterationInfo,
    StoppingDecision
)
from defame.analysis.analyzers.iteration.prompts import (
    get_information_gain_prompt,
    get_stopping_decision_prompt,
)
from defame.analysis.data_models import ClaimData
from defame.analysis.llm_helper import AnalyzerLLMHelper


def extract_iteration_data(
    claim_data: ClaimData,
    llm: AnalyzerLLMHelper | None = None,
    max_iterations_available: int = 5  # Default max, can be configured
) -> IterationExtractedData:
    """Extract iteration data for a single claim."""
    num_iterations = len(claim_data.log.iterations)

    # Extract information about each iteration
    iterations = []
    cumulative_evidence = []  # Track all evidence seen so far

    for i, iteration in enumerate(claim_data.log.iterations):
        # Get evidence from this iteration (from result summaries)
        current_evidence = []
        for action in iteration.evidence_retrieval:
            for result in action.results:
                if result.summary:
                    current_evidence.append(result.summary)

        # Calculate information gain using LLM
        information_gain = 0.0
        if llm is not None and current_evidence:
            evidence_before = cumulative_evidence.copy()
            evidence_after = cumulative_evidence + current_evidence

            prompt = get_information_gain_prompt(
                claim_data.claim_text,
                evidence_before,
                evidence_after
            )
            response = llm.generate(prompt)
            if isinstance(response, str):
                information_gain = _extract_score_from_response(response)

        # Update cumulative evidence
        cumulative_evidence.extend(current_evidence)

        # Create iteration info
        iterations.append(IterationInfo(
            iteration_num=i,
            has_judgement=iteration.judgment.verdict != "",
            information_gain=information_gain,
            num_evidence_pieces=len(current_evidence)
        ))

    # Assess stopping decision using LLM
    stopping_decision = None
    if llm is not None and cumulative_evidence:
        prompt = get_stopping_decision_prompt(
            claim_data.claim_text,
            cumulative_evidence,
            claim_data.ground_truth_label,
            claim_data.correct,
            num_iterations - 1
        )
        response = llm.generate(prompt)
        if isinstance(response, str):
            stopping_decision = _parse_stopping_decision(response)

    return IterationExtractedData(
        claim_id=int(claim_data.claim_id) if claim_data.claim_id.isdigit() else 0,
        claim_text=claim_data.claim_text,
        prediction_correct=claim_data.correct,
        num_iterations=num_iterations,
        max_iterations_available=max_iterations_available,
        iterations=iterations,
        stopping_decision=stopping_decision
    )


def _extract_score_from_response(response: str) -> float:
    """Extract score (1-5) from LLM response."""
    match = re.search(r"(\d+(\.\d+)?)", response)
    if match:
        score = float(match.group(1))
        if 1.0 <= score <= 5.0:
            return score
    return 0.0


def _parse_stopping_decision(response: str) -> StoppingDecision:
    """Parse stopping decision from LLM response."""
    lines = [line.strip() for line in response.strip().split('\n') if line.strip()]

    if not lines:
        return StoppingDecision(decision="Uncertain", reasoning="Unable to parse response")

    # First line should be the decision
    decision = lines[0]

    # Normalize decision
    if "appropriate" in decision.lower() and "should" not in decision.lower():
        decision = "Appropriate"
    elif "should_continue" in decision.lower() or "should continue" in decision.lower():
        decision = "Should_continue"
    else:
        decision = "Uncertain"

    # Rest is reasoning
    reasoning = " ".join(lines[1:]) if len(lines) > 1 else "No reasoning provided"

    return StoppingDecision(decision=decision, reasoning=reasoning)
