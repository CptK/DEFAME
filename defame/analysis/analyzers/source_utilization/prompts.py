"""LLM prompts for Source Utilization and Tool Effectiveness Analyzer."""


def get_reasoning_citation_prompt(
    claim: str, evidence_summaries: list[str], reasoning_text: str
) -> str:
    """
    Prompt to analyze whether reasoning section cites and uses the evidence found.

    Args:
        claim: The claim being fact-checked
        evidence_summaries: List of evidence summaries from useful sources
        reasoning_text: The reasoning/analysis section from elaboration block

    Returns:
        Prompt asking if reasoning cites evidence (Y/N) and rating quality (0-1)
    """
    evidence_list = "\n".join(f"- {ev}" for ev in evidence_summaries)

    return f"""You are analyzing whether a fact-checking reasoning section properly cites and uses the evidence that was gathered.

CLAIM:
{claim}

EVIDENCE GATHERED:
{evidence_list}

REASONING SECTION:
{reasoning_text}

TASK:
1. Determine if the reasoning section cites and uses the evidence gathered. Answer with Y or N.
2. Rate the quality of evidence usage on a 0-1 scale:
   - 0.0: No evidence cited or used
   - 0.5: Some evidence cited but not thoroughly integrated
   - 1.0: Evidence thoroughly cited and well-integrated into reasoning

Provide your response in this EXACT format:
CITES_EVIDENCE: [Y or N]
CITATION_SCORE: [0.0 to 1.0]

Do not include any other text in your response."""


def get_cross_reference_identification_prompt(
    claim: str, useful_results: list[tuple[str, str]]
) -> str:
    """
    Prompt to identify which sources corroborate the same facts.

    Args:
        claim: The claim being fact-checked
        useful_results: List of (url, content) tuples from useful sources

    Returns:
        Prompt asking LLM to identify cross-referenced facts in JSON format
    """
    sources_text = ""
    for idx, (url, content) in enumerate(useful_results, 1):
        sources_text += f"\nSOURCE {idx}:\nURL: {url}\nCONTENT: {content}\n"

    return f"""You are analyzing whether multiple sources corroborate the same facts in a fact-checking context.

CLAIM:
{claim}

SOURCES:
{sources_text}

TASK:
Identify facts that are mentioned by multiple sources (cross-referenced). For each fact that appears in 2+ sources:
1. State the fact clearly
2. List which source numbers support it

Provide your response as a JSON array with this structure:
[
  {{
    "fact": "The specific fact being corroborated",
    "supporting_sources": [1, 3, 5]
  }},
  ...
]

Only include facts that are supported by at least 2 sources. If no facts are cross-referenced, return an empty array: []

Respond with ONLY the JSON array, no additional text."""


def get_fact_clustering_prompt(evidence_texts: list[str]) -> str:
    """
    Alternative prompt for clustering similar evidence statements.

    Args:
        evidence_texts: List of evidence text summaries

    Returns:
        Prompt asking LLM to group similar evidence
    """
    evidence_list = "\n".join(f"{i+1}. {text}" for i, text in enumerate(evidence_texts))

    return f"""You are grouping evidence statements that express similar or related facts.

EVIDENCE STATEMENTS:
{evidence_list}

TASK:
Group the evidence statements that discuss the same fact or closely related facts. Each group should contain statements that corroborate each other.

Provide your response as a JSON array where each element is a group (array of statement numbers):
[
  [1, 3, 5],
  [2, 4],
  [6]
]

Statement numbers that don't relate to any other statements should be in groups by themselves.

Respond with ONLY the JSON array, no additional text."""
