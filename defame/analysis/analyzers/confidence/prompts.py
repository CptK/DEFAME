"""LLM prompts for Confidence Analyzer."""

def get_confidence_strength_prompt(judgement: str) -> str:
    """
    Prompt to rate confidence strength based on judgement.

    Args:
        judgement: The judgement text from the iteration

    Returns:
        Prompt asking for confidence strength rating (1-5)
    """
    return f"""You are given the judgement of fact-checking verdict. Based on this judgement, rate the confidence strength with which the text presents its conclusion.
JUDGEMENT:
{judgement}

TASK:
Rate the confidence strength on a scale from 1 to 5:
1 - Very Low Confidence: The judgement is vague, uncertain, or lacks clarity about the conclusion
2 - Low Confidence: The judgement shows some uncertainty or hesitation about the conclusion
3 - Moderate Confidence: The judgement is somewhat clear but may have minor doubts or ambiguities
4 - High Confidence: The judgement is clear and assertive about the conclusion with minimal doubt
5 - Very High Confidence: The judgement is definitive, unambiguous, and strongly asserts the conclusion.

IMPORTANT:
Provide your response as a single number (1-5) without any additional text.
"""
    

def get_evidence_strength_prompt(
    claim: str, evidence: list[str]
) -> str:
    """
    Prompt to rate evidence strength based on claim and evidence.

    Args:
        claim: The claim being fact-checked
        evidence: List of evidence summaries

    Returns:
        Prompt asking for evidence strength rating (1-5)
    """
    evidence_list = "\n".join(f"- {ev}" for ev in evidence)
    return f"""You are given a claim and a set of evidence summaries used to fact-check it. Based on the quality and relevance of the evidence, rate the strength of the evidence supporting the claim.
CLAIM:
{claim}

EVIDENCE:
{evidence_list}

EVIDENCE STRENGTH (1-5):
Rate the evidence strength on a scale from 1 to 5:
1 - Very Weak Evidence: The evidence is irrelevant, unreliable, or insufficient to make a judgement about the claim.
2 - Weak Evidence: The evidence has limited relevance or reliability, providing minimal information for evaluating the claim.
3 - Moderate Evidence: The evidence is somewhat relevant and reliable, offering a reasonable basis for evaluating the claim.
4 - Strong Evidence: The evidence is relevant, reliable, and provides a solid foundation for evaluating the claim.
5 - Very Strong Evidence: The evidence is highly relevant, credible, and comprehensive, providing an excellent basis for evaluating the claim.

IMPORTANT:
Provide your response as a single number (1-5) without any additional text.
"""


def get_hedging_prompt(judgement: str) -> str:
    """
    Prompt to rate hedging language in judgement.

    Args:
        judgement: The judgement text from the iteration

    Returns:
        Prompt asking for hedging score rating (1-5)
    """
    return f"""You are given the judgement of a fact-checking verdict. Rate the degree of hedging language used in expressing the conclusion.

JUDGEMENT:
{judgement}

TASK:
Rate the hedging language on a scale from 1 to 5:
1 - No Hedging: The judgement uses definitive, unqualified language (e.g., "is false", "is true", "clearly shows")
2 - Minimal Hedging: The judgement mostly uses direct language with rare qualifiers (e.g., "appears to be", "seems")
3 - Moderate Hedging: The judgement uses some hedging language (e.g., "suggests", "likely", "may", "could")
4 - Significant Hedging: The judgement frequently uses hedging language (e.g., "might", "possibly", "uncertain", "unclear")
5 - Heavy Hedging: The judgement is highly tentative with extensive use of hedging (e.g., "it's difficult to say", "cannot be certain", "remains unclear")

IMPORTANT:
Provide your response as a single number (1-5) without any additional text.
"""
