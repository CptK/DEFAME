"""LLM prompts for Iteration & Stopping Criteria Analyzer."""


def get_information_gain_prompt(
    claim: str,
    evidence_before: list[str],
    evidence_after: list[str]
) -> str:
    """
    Prompt to rate information gain from an iteration.

    Args:
        claim: The claim being fact-checked
        evidence_before: Evidence summaries from previous iterations
        evidence_after: All evidence summaries including new iteration

    Returns:
        Prompt asking for information gain rating (1-5)
    """
    # Get only the new evidence (what was added in this iteration)
    new_evidence = evidence_after[len(evidence_before):]

    evidence_before_str = "\n".join(f"- {ev}" for ev in evidence_before) if evidence_before else "(No previous evidence)"
    new_evidence_str = "\n".join(f"- {ev}" for ev in new_evidence) if new_evidence else "(No new evidence)"

    return f"""You are evaluating how much new, relevant information was discovered in a fact-checking iteration.

CLAIM:
{claim}

EVIDENCE BEFORE THIS ITERATION:
{evidence_before_str}

NEW EVIDENCE FROM THIS ITERATION:
{new_evidence_str}

TASK:
Rate the information gain on a scale from 1 to 5:
1 - No Gain: No new relevant information discovered, or evidence is redundant/irrelevant
2 - Minimal Gain: Very little new information, mostly repeats what was already known
3 - Moderate Gain: Some new relevant information that adds to understanding
4 - Significant Gain: Substantial new information that meaningfully advances the investigation
5 - Major Gain: Critical new evidence that dramatically changes understanding or enables a verdict

IMPORTANT:
Provide your response as a single number (1-5) without any additional text.
"""


def get_stopping_decision_prompt(
    claim: str,
    all_evidence: list[str],
    ground_truth_label: str,
    prediction_correct: bool,
    iteration_num: int
) -> str:
    """
    Prompt to assess whether stopping was appropriate.

    Args:
        claim: The claim being fact-checked
        all_evidence: All evidence summaries collected
        ground_truth_label: The true/correct label
        prediction_correct: Whether the prediction was correct
        iteration_num: The iteration number where it stopped

    Returns:
        Prompt asking whether stopping was appropriate
    """
    evidence_str = "\n".join(f"- {ev}" for ev in all_evidence) if all_evidence else "(No evidence collected)"

    if prediction_correct:
        outcome_str = f"""GROUND TRUTH LABEL: {ground_truth_label}
PREDICTION OUTCOME: ✓ CORRECT"""
        instruction = """Consider:
- Is the evidence sufficient to support reaching the correct conclusion ({ground_truth_label})?
- Could they have reached this conclusion more efficiently (with less evidence)?
- Is the correct prediction well-supported by evidence, or did they get lucky?"""
    else:
        outcome_str = f"""GROUND TRUTH LABEL: {ground_truth_label}
PREDICTION OUTCOME: ✗ WRONG - They did NOT predict the correct label"""
        instruction = f"""Consider:
- Does the evidence mislead away from the truth, or is it simply incomplete?
- Would additional searching likely have found evidence supporting the correct label ({ground_truth_label})?
- Are there obvious gaps that, if filled, would have led to the correct answer?
- Did they stop prematurely before finding critical evidence for the truth?"""

    return f"""You are evaluating whether a fact-checking process stopped at the right time.

CLAIM:
{claim}

ALL EVIDENCE COLLECTED (across {iteration_num + 1} iteration(s)):
{evidence_str}

{outcome_str}

TASK:
Assess whether stopping at this point was appropriate, or if they should have continued searching for more evidence.

{instruction}

Respond with ONLY ONE of these options:
- Appropriate: The stopping decision was reasonable (correct prediction with sufficient evidence, OR wrong prediction where evidence was genuinely misleading and more searching wouldn't help)
- Should_continue: They stopped too early (especially for wrong predictions where more evidence would likely have revealed the truth)
- Uncertain: It's unclear whether stopping was appropriate

IMPORTANT:
Respond with ONLY the option name (Appropriate, Should_continue, or Uncertain) on the first line.
Then provide a brief 1-2 sentence explanation on the second line.

Format:
[Decision]
[Brief explanation]
"""
