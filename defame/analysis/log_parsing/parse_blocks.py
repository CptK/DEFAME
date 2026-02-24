"""
Parser for iteration blocks.

Takes the raw text blocks from split_into_blocks.py and parses them
directly into StructuredLog components (Pydantic models).
"""

import re
from defame.analysis.log_parsing import IterationBlocks
from defame.common.structured_log_schema import (
    PlanningInfo,
    ElaborationInfo,
    JudgmentInfo,
    EvidenceRetrievalAction,
    ResultInfo,
    IterationInfo,
)


def _placeholder_timestamp() -> str:
    """Return empty string as placeholder for missing timestamps."""
    return ""


def parse_planning_block(planning_text: str | None) -> PlanningInfo:
    """
    Parse the planning block to extract proposed actions.

    Args:
        planning_text: Raw planning block text

    Returns:
        PlanningInfo object
    """
    if not planning_text:
        return PlanningInfo(
            timestamp=_placeholder_timestamp(),
            plan_text="",
        )

    actions_planned = []

    # Extract code blocks
    code_blocks = re.findall(r'```(.*?)```', planning_text, re.DOTALL)

    # Extract action calls from code blocks
    for block in code_blocks:
        # Match patterns like: web_search("query") or image_search("query")
        action_pattern = r'(\w+_search)\(["\']([^"\']+)["\']\)'
        matches = re.findall(action_pattern, block)

        for tool_name, query in matches:
            actions_planned.append(f"{tool_name}({query})")

    return PlanningInfo(
        timestamp=_placeholder_timestamp(),
        plan_text=planning_text,
        actions_planned=actions_planned,
    )


def parse_actions_and_evidence_block(actions_evidence_text: str | None) -> list[EvidenceRetrievalAction]:
    """
    Parse the actions and evidence block to extract execution logs, results, and summaries.

    Args:
        actions_evidence_text: Raw actions and evidence block text

    Returns:
        List of EvidenceRetrievalAction objects
    """
    if not actions_evidence_text:
        return []

    # 1. Extract search executions ("Searching" lines)
    search_executions = []
    search_pattern = r'Searching (\S+) with query: (.*?)(?=\n)'
    for match in re.finditer(search_pattern, actions_evidence_text):
        platform = match.group(1)
        full_query = match.group(2)

        # Try to extract the text query from TextQuery(...)
        text_match = re.search(r"text=['\"]([^'\"]+)['\"]", full_query)
        query = text_match.group(1) if text_match else full_query

        search_executions.append({
            'platform': platform,
            'query': query,
            'full_query_repr': full_query
        })

    # 2. Extract result blocks ("Got X new source(s):")
    result_blocks = []
    result_pattern = r'Got (\d+) new (?:web )?source\(s\):(.*?)(?=\nGot \d+|Generated response:|\nSearching |\Z)'
    for match in re.finditer(result_pattern, actions_evidence_text, re.DOTALL):
        count = int(match.group(1))
        urls_text = match.group(2)

        # Extract URLs (handle optional date prefixes)
        url_pattern = r'\d+\.\s+(?:(.*?\d{4})\s+)?(https?://\S+)'
        urls = []
        for url_match in re.finditer(url_pattern, urls_text):
            url = url_match.group(2)
            urls.append(url)

        result_blocks.append({'count': count, 'urls': urls})

    # 3. Extract useful results
    useful_results = {}
    useful_pattern = r'Useful result: From \[Source\]\((https?://[^\)]+)\):\s*\nContent: (.*?)(?=\nGenerated response:|\nUseful result:|\nSearching |\Z)'
    for match in re.finditer(useful_pattern, actions_evidence_text, re.DOTALL):
        url = match.group(1)
        summary = match.group(2).strip()
        useful_results[url] = summary

    # 4. Extract errors and warnings
    errors = []
    error_patterns = [
        r'(Skipping fact-checking website: [^\n]+)',
        r'(Skipping unsupported website: [^\n]+)',
        r'(Failed to scrape [^\n]+)',
        r'(Error \d+: [^\n]+)',
        r'(Failed to download or open image[^\n]+)',
        r'(Missed new sources!)',
    ]
    for pattern in error_patterns:
        errors.extend(re.findall(pattern, actions_evidence_text))

    # Create EvidenceRetrievalAction objects by grouping results with their searches
    actions: list[EvidenceRetrievalAction] = []

    for i, search_exec in enumerate(search_executions):
        # Try to find matching result block
        result_info_list: list[ResultInfo] = []

        if i < len(result_blocks):
            result_block = result_blocks[i]

            # Convert URLs to ResultInfo objects
            for url in result_block['urls']:
                # Check if this URL was marked as useful
                marked_useful = url in useful_results
                summary = useful_results.get(url)

                result_info = ResultInfo(
                    url=url,
                    timestamp=_placeholder_timestamp(),
                    marked_useful=marked_useful,
                    summary=summary,
                )
                result_info_list.append(result_info)

        # Create the action
        action = EvidenceRetrievalAction(
            action_type="search",
            tool=f"{search_exec['platform']}_search",
            timestamp=_placeholder_timestamp(),
            results=result_info_list,
            total_results=len(result_info_list),
            unique_results=len(result_info_list),
            errors=errors if i == 0 else [],  # Put errors on first action
            query=search_exec['query'],
            platform=search_exec['platform'],
        )
        actions.append(action)

    # If no search executions but we have result blocks, create generic actions
    if not search_executions and result_blocks:
        for result_block in result_blocks:
            result_info_list = [
                ResultInfo(
                    url=url,
                    timestamp=_placeholder_timestamp(),
                    marked_useful=url in useful_results,
                    summary=useful_results.get(url),
                )
                for url in result_block['urls']
            ]

            action = EvidenceRetrievalAction(
                action_type="search",
                tool="unknown",
                timestamp=_placeholder_timestamp(),
                results=result_info_list,
                total_results=len(result_info_list),
                unique_results=len(result_info_list),
            )
            actions.append(action)

    return actions


def parse_elaboration_block(elaboration_text: str | None) -> ElaborationInfo:
    """
    Parse the elaboration block.

    Args:
        elaboration_text: Raw elaboration block text

    Returns:
        ElaborationInfo object
    """
    if not elaboration_text:
        return ElaborationInfo(
            timestamp=_placeholder_timestamp(),
            analysis_text="",
        )

    return ElaborationInfo(
        timestamp=_placeholder_timestamp(),
        analysis_text=elaboration_text,
    )


def parse_judgement_block(judgement_text: str | None) -> JudgmentInfo:
    """
    Parse the judgement block to extract the decision.

    Args:
        judgement_text: Raw judgement block text

    Returns:
        JudgmentInfo object
    """
    if not judgement_text:
        return JudgmentInfo(
            timestamp=_placeholder_timestamp(),
            reasoning="",
            verdict="",
        )

    # Try to find decision mentions (multiple formats)
    # Format 1: In backticks like `not enough information`
    # Format 2: "is <verdict>" like "is conflicting evidence"
    # Format 3: "veracity is: <verdict>" like "veracity is: conflicting evidence"
    decision_patterns = [
        # Backtick format
        r'`(not enough information)`',
        r'`(conflicting evidence)`',
        r'`(supported)`',
        r'`(refuted)`',
        r'`(true)`',
        r'`(false)`',
        r'`(nei)`',
        # "is <verdict>" format
        r'is\s+(conflicting evidence)',
        r'is\s+(supported)',
        r'is\s+(refuted)',
        r'is\s+(not enough information)',
        r'is\s+(nei)',
        # "veracity is: <verdict>" format
        r'veracity is:\s+(conflicting evidence)',
        r'veracity is:\s+(supported)',
        r'veracity is:\s+(refuted)',
        r'veracity is:\s+(not enough information)',
        r'veracity is:\s+(nei)',
    ]

    verdict = ""
    for pattern in decision_patterns:
        match = re.search(pattern, judgement_text, re.IGNORECASE)
        if match:
            verdict = match.group(1).upper().replace(' ', '_')
            break

    return JudgmentInfo(
        timestamp=_placeholder_timestamp(),
        reasoning=judgement_text,
        verdict=verdict,
    )


def parse_iteration_blocks(blocks: IterationBlocks, iteration_number: int) -> IterationInfo:
    """
    Parse all blocks in an iteration into an IterationInfo object.

    Args:
        blocks: IterationBlocks from split_into_blocks.py
        iteration_number: Iteration number (1-indexed)

    Returns:
        IterationInfo with structured data
    """
    return IterationInfo(
        iteration_number=iteration_number,
        timestamp_start=_placeholder_timestamp(),
        timestamp_end=_placeholder_timestamp(),
        planning=parse_planning_block(blocks.planning),
        evidence_retrieval=parse_actions_and_evidence_block(blocks.actions_and_evidence),
        elaboration=parse_elaboration_block(blocks.elaboration),
        judgment=parse_judgement_block(blocks.judgement),
    )


if __name__ == "__main__":
    from defame.analysis.log_parsing import process_log_with_blocks

    # Example usage
    log_path = "/mnt/vast/workspaces/PI_Rohrbach/mk79honu/DEFAME/out/averitec/summary/dynamic/llama4_scout/2025-05-27_06-47 defame/fact-checks/0/log.txt"

    print(f"Processing: {log_path}\n")
    all_iterations = process_log_with_blocks(log_path)

    print(f"Found {len(all_iterations)} iterations\n")
    print("="*70)

    for i, blocks in enumerate(all_iterations, 1):
        print(f"\nITERATION {i}")
        print("-"*70)

        # Parse the blocks
        iteration_info = parse_iteration_blocks(blocks, i)

        # Display parsed planning
        print(f"\n[PLANNING]")
        print(f"  Actions planned: {len(iteration_info.planning.actions_planned)}")
        for j, action in enumerate(iteration_info.planning.actions_planned, 1):
            print(f"    {j}. {action}")

        # Display parsed actions & evidence
        print(f"\n[EVIDENCE RETRIEVAL]")
        print(f"  Total actions: {len(iteration_info.evidence_retrieval)}")
        for j, action in enumerate(iteration_info.evidence_retrieval, 1):
            print(f"    {j}. {action.tool} on {action.platform}")
            print(f"       Query: \"{action.query}\"")
            print(f"       Results: {action.total_results}")
            print(f"       Useful results: {sum(1 for r in action.results if r.marked_useful)}")

        # Display parsed elaboration
        print(f"\n[ELABORATION]")
        print(f"  Text length: {len(iteration_info.elaboration.analysis_text)} chars")

        # Display parsed judgement
        print(f"\n[JUDGMENT]")
        print(f"  Verdict: {iteration_info.judgment.verdict}")
        print(f"  Reasoning length: {len(iteration_info.judgment.reasoning)} chars")

    print("\n" + "="*70)
