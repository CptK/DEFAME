"""
Simple iteration block splitter.

Splits each iteration into text blocks without parsing:
- Planning block (the code blocks with proposed actions)
- Actions & Evidence block (everything between planning and elaboration)
- Elaboration block (the analysis/synthesis section)
- Judgement block (the final decision section)
"""

import re
from dataclasses import dataclass
from pathlib import Path

from defame.analysis.log_parsing import parse_log_iterations


@dataclass
class IterationBlocks:
    """Raw text blocks from an iteration."""
    planning: str | None = None
    actions_and_evidence: str | None = None
    elaboration: str | None = None
    judgement: str | None = None


def split_iteration_into_blocks(iteration_content: str) -> IterationBlocks:
    """
    Split an iteration into its main text blocks using structural markers.

    Returns raw text for each block without parsing.

    Args:
        iteration_content: Full text of one iteration

    Returns:
        IterationBlocks with raw text for each block
    """
    blocks = IterationBlocks()

    # Clean up the iteration marker
    content = iteration_content.replace("Not enough information yet. Continuing fact-check...", "").strip()

    # Find all "Generated response:" positions to use as boundaries
    generated_response_pattern = r"Generated response:"
    response_positions = [(m.start(), m.end()) for m in re.finditer(generated_response_pattern, content)]

    if not response_positions:
        # No responses found, return everything as actions_and_evidence
        blocks.actions_and_evidence = content
        return blocks

    # 1. Extract Planning Block
    # Planning is the first "Generated response:" that contains code blocks (```)
    planning_end = 0
    for i, (start_pos, end_pos) in enumerate(response_positions):
        # Look for the end of this response (marked by <|eot|>)
        eot_match = re.search(r"<\|eot\|>", content[end_pos:])
        if eot_match:
            response_end = end_pos + eot_match.end()
            response_text = content[start_pos:response_end]

            # Check if this response contains code blocks
            if "```" in response_text:
                blocks.planning = response_text
                planning_end = response_end
                break

    if planning_end == 0:
        # No planning with code blocks found, return everything as actions_and_evidence
        blocks.actions_and_evidence = content
        return blocks

    # 2. Find where elaboration starts
    # Elaboration typically starts with a "Generated response:" containing "## Analysis"
    # But it must come AFTER the evidence collection phase

    # First, find the last evidence marker (either "Useful result:" or "NONE<|eot|>")
    evidence_markers = [
        (m.end(), "useful") for m in re.finditer(r"Useful result:", content[planning_end:])
    ] + [
        (m.end(), "none") for m in re.finditer(r"Generated response:\s*NONE<\|eot\|>", content[planning_end:])
    ]

    if not evidence_markers:
        # No evidence found, everything after planning is actions_and_evidence
        blocks.actions_and_evidence = content[planning_end:].strip()
        return blocks

    # Sort by position and take the last one
    evidence_markers.sort(key=lambda x: x[0])
    last_evidence_pos = planning_end + evidence_markers[-1][0]

    # Now find the next "Generated response:" after the last evidence
    # This should be the elaboration
    elaboration_start = None
    for start_pos, end_pos in response_positions:
        if start_pos > last_evidence_pos:
            elaboration_start = start_pos
            break

    if not elaboration_start:
        # No elaboration found
        blocks.actions_and_evidence = content[planning_end:].strip()
        return blocks

    # 3. Split actions_and_evidence from elaboration+judgement
    blocks.actions_and_evidence = content[planning_end:elaboration_start].strip()

    # 4. Find where judgement starts within elaboration+judgement
    # Judgement contains decision-making keywords
    remaining_text = content[elaboration_start:]
    decision_keywords = [
        'key insights from the fact-check',
        'decision option',
        'applies best',
        'the decision option that applies best'
    ]

    judgement_start = None
    for start_pos, end_pos in response_positions:
        if start_pos >= elaboration_start:
            # Check if this response contains decision keywords
            eot_match = re.search(r"<\|eot\|>", content[end_pos:])
            if eot_match:
                response_end = end_pos + eot_match.end()
                response_text = content[start_pos:response_end].lower()

                if any(keyword in response_text for keyword in decision_keywords):
                    judgement_start = start_pos
                    break

    if judgement_start:
        blocks.elaboration = content[elaboration_start:judgement_start].strip()
        blocks.judgement = content[judgement_start:].strip()
    else:
        # No judgement found, everything is elaboration
        blocks.elaboration = remaining_text.strip()

    return blocks


def process_log_with_blocks(log_path: str | Path) -> list[IterationBlocks]:
    """Parse a log file into iterations and then into blocks."""
    iterations = parse_log_iterations(log_path)
    return [split_iteration_into_blocks(it) for it in iterations]


def print_block_summary(blocks: IterationBlocks):
    """Print a simple summary of the blocks."""
    print("\nBlocks found:")
    print(f"  Planning: {'✓' if blocks.planning else '✗'} ({len(blocks.planning) if blocks.planning else 0} chars)")
    print(f"  Actions & Evidence: {'✓' if blocks.actions_and_evidence else '✗'} ({len(blocks.actions_and_evidence) if blocks.actions_and_evidence else 0} chars)")
    print(f"  Elaboration: {'✓' if blocks.elaboration else '✗'} ({len(blocks.elaboration) if blocks.elaboration else 0} chars)")
    print(f"  Judgement: {'✓' if blocks.judgement else '✗'} ({len(blocks.judgement) if blocks.judgement else 0} chars)")


if __name__ == "__main__":
    # Example usage
    log_path = "/mnt/vast/workspaces/PI_Rohrbach/mk79honu/DEFAME/out/mocheg/summary/dynamic/llama4_scout/0-100/fact-checks/6/log.txt"
    log_path = "/mnt/vast/workspaces/PI_Rohrbach/mk79honu/DEFAME/out/averitec/summary/dynamic/llama4_scout/2025-05-27_06-47 defame/fact-checks/0/log.txt"

    print(f"Processing: {log_path}\n")
    all_iterations = process_log_with_blocks(log_path)

    print(f"Found {len(all_iterations)} iterations\n")
    print("="*60)

    for i, blocks in enumerate(all_iterations, 1):
        print(f"\nITERATION {i}")
        print("-"*60)
        print_block_summary(blocks)

        # Show a preview of each block
        if blocks.planning:
            print("\n  Planning preview:")
            print(f"    {blocks.planning[:200]}...")

        if blocks.actions_and_evidence:
            print("\n  Actions & Evidence preview:")
            print(f"    {blocks.actions_and_evidence[:200]}...")

        if blocks.elaboration:
            print("\n  Elaboration preview:")
            print(f"    {blocks.elaboration[:200]}...")

        if blocks.judgement:
            print("\n  Judgement preview:")
            print(f"    {blocks.judgement[:200]}...")

    print("\n" + "="*60)
