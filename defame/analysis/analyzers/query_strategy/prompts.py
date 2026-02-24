QUERY_SPECIFICITY_PROMPT = """Rate this fact-checking search query's specificity from 1-5, where:
1 = Very vague (e.g., "Biden news")
2 = Somewhat vague (e.g., "Biden age")
3 = Moderately specific (e.g., "How old is Joe Biden?")
4 = Quite specific (e.g., "Joe Biden age in 2024")
5 = Very specific (e.g., "Joe Biden's exact age on January 20, 2024")

Query: "{query}"

Output only a single number from 1-5. No explanation."""


COUNTER_EVIDENCE_SEEKING_PROMPT = """Does this search query seek counter-evidence or alternative perspectives?

A query seeks counter-evidence if it:
- Questions the validity of a claim
- Looks for opposing viewpoints
- Searches for fact-checks or debunking information
- Uses skeptical language

Query: "{query}"

Answer with only "Yes" or "No"."""


GROUP_QUERIES_BY_ANGLE_PROMPT = """Given this claim being fact-checked:
"{claim}"

These search queries were used:
{queries_str}

Identify the distinct aspects/angles being investigated. Group the queries by topic.

Output format (JSON):
{{
    "Angle 1 name": [1, 3, 5],
    "Angle 2 name": [2, 4],
    "Angle 3 name": [6]
}}

Where numbers refer to query indices. Output only the JSON, no explanation."""


TOOL_COHERENCE_RATING_PROMPT = """Evaluate if the next action logically follows from previous evidence in a fact-checking process.

Previous action: {previous_action}
Previous evidence: {previous_evidence}
Next action: {next_action}

Does the next action logically follow from the previous evidence?
Rate from 1-5 where:
1 = Not at all coherent (completely unrelated)
2 = Slightly coherent (weak connection)
3 = Moderately coherent (some logical connection)
4 = Quite coherent (clear logical connection)
5 = Highly coherent (strongly follows from evidence)

Output only a single number from 1-5. No explanation."""