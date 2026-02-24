"""LLM prompts for Reasoning Quality Analyzer."""


def get_logical_coherence_prompt(claim: str, reasoning: str) -> str:
    """
    Prompt to rate logical coherence of reasoning.

    Args:
        claim: The claim being fact-checked
        reasoning: The reasoning text to analyze

    Returns:
        Prompt asking for coherence rating (1-5) and explanation
    """
    return f"""You are analyzing the logical coherence of reasoning in a fact-checking context.

CLAIM:
{claim}

REASONING:
{reasoning}

TASK:
Rate the logical coherence of the reasoning on a 1-5 scale:

1 - Incoherent: Reasoning is disjointed, contradictory, or makes no logical sense
2 - Poor: Multiple logical gaps, weak connections between ideas
3 - Acceptable: Basic logical flow but with some gaps or unclear connections
4 - Good: Clear logical flow with minor issues
5 - Excellent: Highly coherent, each step follows logically from the previous

Consider:
- Are ideas connected logically?
- Are there contradictions?
- Does each step follow from the previous?
- Are conclusions supported by the preceding analysis?

Provide your response in this EXACT format:
COHERENCE_SCORE: [1-5]
EXPLANATION: [1-2 sentences explaining the rating]

Do not include any other text in your response."""


def get_evidence_claim_connection_prompt(
    claim: str, evidence: list[str], reasoning: str
) -> str:
    """
    Prompt to assess if reasoning addresses the claim and uses evidence.

    Args:
        claim: The claim being fact-checked
        evidence: List of evidence summaries
        reasoning: The reasoning text to analyze

    Returns:
        Prompt asking if reasoning addresses claim (Y/N) and connection strength (0-1)
    """
    evidence_list = "\n".join(f"- {ev}" for ev in evidence)

    return f"""You are analyzing whether reasoning properly addresses the claim using the available evidence.

CLAIM:
{claim}

EVIDENCE AVAILABLE:
{evidence_list}

REASONING:
{reasoning}

TASK:
1. Does the reasoning actually address the specific claim being fact-checked? (Y/N)
2. Rate the strength of the evidence-to-claim connection (0-1 scale):
   - 0.0: No connection, reasoning doesn't use evidence or address claim
   - 0.3: Weak connection, mentions evidence but doesn't tie to claim
   - 0.5: Moderate connection, uses some evidence to address parts of claim
   - 0.7: Strong connection, systematically uses evidence to address claim
   - 1.0: Excellent connection, thoroughly integrates all relevant evidence to directly address claim

Provide your response in this EXACT format:
ADDRESSES_CLAIM: [Y or N]
CONNECTION_STRENGTH: [0.0 to 1.0]
EXPLANATION: [1-2 sentences explaining the assessment]

Do not include any other text in your response."""


def get_logical_chain_prompt(claim: str, evidence: list[str], reasoning: str) -> str:
    """
    Prompt to trace and evaluate the logical chain from evidence to conclusion.

    Args:
        claim: The claim being fact-checked
        evidence: List of evidence summaries
        reasoning: The reasoning text to analyze

    Returns:
        Prompt asking for chain strength rating and identification of breaks
    """
    evidence_list = "\n".join(f"{i+1}. {ev}" for i, ev in enumerate(evidence))

    return f"""You are tracing the logical chain from evidence through reasoning to conclusion.

CLAIM:
{claim}

EVIDENCE:
{evidence_list}

REASONING:
{reasoning}

TASK:
Trace the logical chain: Evidence A → Inference B → Conclusion C

1. Is each step justified by the previous?
2. Are there logical gaps or leaps?
3. Does the chain flow logically from evidence to inference to conclusion?

Rate the chain strength (1-5):
1 - Broken chain: Major leaps, unjustified conclusions, missing links
2 - Weak chain: Several gaps, some unjustified steps
3 - Acceptable chain: Generally follows but with some gaps
4 - Strong chain: Clear progression with minor gaps
5 - Excellent chain: Each step clearly justified, tight logical flow

Identify any breaks in the chain (places where logic jumps without justification).

Provide your response in this EXACT format:
CHAIN_STRENGTH: [1-5]
CHAIN_BREAKS: [List each break as a bullet point, or write "None" if no breaks]
EXPLANATION: [1-2 sentences explaining the chain analysis]

Do not include any other text in your response."""


def get_synthesis_quality_prompt(
    claim: str, evidence: list[str], reasoning: str
) -> str:
    """
    Prompt to evaluate synthesis quality across multiple sources.

    Args:
        claim: The claim being fact-checked
        evidence: List of evidence summaries from different sources
        reasoning: The reasoning text to analyze

    Returns:
        Prompt asking for synthesis quality rating (1-5)
    """
    evidence_list = "\n".join(f"{i+1}. {ev}" for i, ev in enumerate(evidence))
    num_sources = len(evidence)

    return f"""You are evaluating how well the reasoning synthesizes information across multiple sources.

CLAIM:
{claim}

EVIDENCE FROM {num_sources} SOURCE(S):
{evidence_list}

REASONING:
{reasoning}

TASK:
Evaluate the synthesis quality (1-5):

1 - No synthesis: Treats sources in isolation or ignores multiple sources
2 - Poor synthesis: Mentions multiple sources but doesn't integrate them
3 - Acceptable synthesis: Some integration but superficial or incomplete
4 - Good synthesis: Effectively integrates sources, identifies patterns
5 - Excellent synthesis: Sophisticated integration, identifies agreements/disagreements, builds coherent narrative

Consider:
- Does reasoning integrate information across sources?
- Are patterns or commonalities identified?
- Are contradictions between sources addressed?
- Is a coherent narrative built from multiple perspectives?

Provide your response in this EXACT format:
SYNTHESIS_SCORE: [1-5]
EXPLANATION: [1-2 sentences explaining the rating]

Do not include any other text in your response."""


def get_logical_fallacies_prompt(
    claim: str,
    reasoning: str,
    iteration_num: int = 0,
    is_final_iteration: bool = False,
    prediction_correct: bool | None = None
) -> str:
    
    iteration_context = (
        "Final iteration" if is_final_iteration 
        else f"Exploratory iteration {iteration_num}"
    )
    
    outcome_context = ""
    if is_final_iteration and prediction_correct is not None:
        outcome_context = f" - Prediction: {'CORRECT' if prediction_correct else 'INCORRECT'}"
    
    return f"""Identify logical fallacies in fact-checking reasoning.

CONTEXT: {iteration_context}{outcome_context}

CLAIM: {claim}

REASONING: {reasoning}

================================================================================
IDENTIFY FALLACIES FROM THESE 5 CATEGORIES:

1. INSUFFICIENT EVIDENCE
   Drawing conclusions without adequate support
   
   Flag when:
   • "One source says X, so X is universally true" (limited sources)
   • "No evidence found for X (personal experience), so X didn't happen" (unknowable)
   
   Don't flag when:
   • Multiple (3+) independent sources cited
   • Checking official records that would contain the event (government, legal, published)

2. FAULTY CAUSATION
   Incorrect causal reasoning
   
   Flag when:
   • "A happened, then B happened, so A caused B" (temporal sequence only)
   • "If we allow X, then Y will inevitably happen" (unjustified chain)
   
   Don't flag when:
   • Just establishing timeline without claiming causation
   • Evidence beyond timing supports causation

3. MISREPRESENTATION  
   Changing or distorting the original claim
   
   Flag when:
   • Claim: "X supports Y" but reasoning addresses "X created Y" (strawman)
   • Using key terms with shifting meanings (equivocation)
   
   Don't flag when:
   • Clarifying genuinely ambiguous wording
   • Making necessary distinctions

4. FALSE DICHOTOMY
   Forcing binary choices when nuance exists
   
   Flag when:
   • "Either completely true or completely false" (ignoring middle ground)
   • Presenting only 2 options when more exist
   
   Don't flag when:
   • Reasoning acknowledges partial truth or complexity

5. IRRELEVANCE
   Logic that doesn't address the actual claim
   
   Flag when:
   • Conclusion restates premise with no new information (circular)
   • Extended discussion of topics not in the claim (distraction)
   
   Don't flag when:
   • Providing context mentioned in original claim
   • Background information that informs the claim

================================================================================
GUIDELINES:

- Expect to find fallacies regularly - they are common in reasoning
- Focus on clear errors that weaken the logical chain
- Standard fact-checking (citing sources, checking records) is NOT a fallacy
- If prediction was correct, be more conservative

================================================================================
CRITICAL CONTEXT: 

When evaluating reasoning from a fact-checking system,  certain types of events are systematically documented in official records:

- Government/Legislative records (bills passed, votes, laws)
- Published media (Forbes lists, magazine rankings, newspaper articles)
- Election results and government appointments
- Court cases and legal proceedings  
- Government alerts and official announcements

When reasoning checks these official records and finds no evidence, this is NOT "Appeal to Ignorance" or "Insufficient Evidence" - it is proper verification using documented sources.
================================================================================
OUTPUT (JSON only):

[
  {{
    "fallacy_type": "Name from above 5 categories",
    "description": "Brief explanation"
  }}
]

Return [] if no fallacies found."""