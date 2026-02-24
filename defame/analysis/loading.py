"""
Module for loading experiment data from disk.

Supports loading from structured JSON logs (preferred) with fallback to text log parsing.
"""

import json
import pandas as pd
from pathlib import Path
import warnings

from defame.analysis.data_models import ClaimData, ExperimentData
from defame.analysis.log_parsing import process_log_with_blocks, parse_iteration_blocks
from defame.common.structured_log_schema import (
    StructuredLog,
    ClaimMetadata,
    FinalVerdict,
    Statistics,
    SCHEMA_VERSION,
)


def load_structured_log_from_json(structured_log_path: Path) -> StructuredLog | None:
    """
    Load a structured log from JSON file.

    Args:
        structured_log_path: Path to structured_log.json

    Returns:
        StructuredLog object or None if loading fails
    """
    try:
        with open(structured_log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return StructuredLog(**data)
    except Exception as e:
        print(f"Warning: Failed to load structured log from {structured_log_path}: {e}")
        return None


def load_structured_log_from_text(
    claim_dir: Path,
    claim_id: str,
    claim_text: str,
    dataset: str,
    predicted_label: str,
) -> StructuredLog | None:
    """
    Load and parse text log, converting to StructuredLog format.

    Args:
        claim_dir: Directory containing the claim's logs
        claim_id: Claim ID
        claim_text: Claim text
        dataset: Dataset name
        predicted_label: Predicted label

    Returns:
        StructuredLog object or None if parsing fails
    """
    log_file = claim_dir / "log.txt"

    if not log_file.exists():
        print(f"Warning: log.txt not found in {claim_dir}")
        return None

    try:
        # Parse text log into IterationInfo objects
        iterations_blocks = process_log_with_blocks(log_file)
        iterations = [
            parse_iteration_blocks(blocks, i + 1)
            for i, blocks in enumerate(iterations_blocks)
        ]

        # Create claim metadata
        claim = ClaimMetadata(
            id=claim_id,
            text=claim_text,
            dataset=dataset,
        )

        # Extract final verdict from last iteration's judgment
        final_verdict_label = predicted_label
        final_verdict_justification = ""

        if iterations and iterations[-1].judgment:
            final_verdict_label = iterations[-1].judgment.verdict or predicted_label
            final_verdict_justification = iterations[-1].judgment.reasoning

        final_verdict = FinalVerdict(
            label=final_verdict_label,
            justification=final_verdict_justification,
            timestamp="",
        )

        # Calculate basic statistics
        total_searches = sum(len(it.evidence_retrieval) for it in iterations)
        total_results = sum(
            action.total_results for it in iterations for action in it.evidence_retrieval
        )
        unique_sources = len(
            set(
                result.url
                for it in iterations
                for action in it.evidence_retrieval
                for result in action.results
            )
        )

        statistics = Statistics(
            total_iterations=len(iterations),
            total_searches=total_searches,
            total_results=total_results,
            unique_sources=unique_sources,
            execution_time_seconds=0.0,  # Not available from text logs
        )

        # Create the structured log
        return StructuredLog(
            version=SCHEMA_VERSION,
            timestamp_start="",
            timestamp_end="",
            claim=claim,
            iterations=iterations,
            final_verdict=final_verdict,
            statistics=statistics,
        )

    except Exception as e:
        print(f"Warning: Failed to parse text log for claim {claim_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_experiment_logs(experiment_dir: str | Path) -> ExperimentData:
    """
    Load experiment data, preferring structured JSON logs with fallback to text parsing.

    This function tries to load structured_log.json for each claim. If not available,
    it falls back to parsing log.txt and converting to StructuredLog format.

    Args:
        experiment_dir: Path to the experiment directory

    Returns:
        ExperimentData object containing all claim data
    """
    if isinstance(experiment_dir, str):
        experiment_dir = Path(experiment_dir)

    predictions = pd.read_csv(experiment_dir / "predictions.csv", index_col="sample_index")

    # Infer dataset name from experiment directory structure or metadata
    dataset = experiment_dir.name

    claims: list[ClaimData] = []

    for claim_dir in (experiment_dir / "fact-checks").iterdir():
        if not claim_dir.is_dir():
            continue

        claim_id = int(claim_dir.name)
        if claim_id not in predictions.index:
            print(f"Warning: claim id {claim_id} not found in predictions.csv, skipping")
            continue

        # Get prediction data
        claim_text = str(predictions.loc[claim_id, "claim"])
        predicted_label = str(predictions.loc[claim_id, "predicted"])
        ground_truth_label = str(predictions.loc[claim_id, "target"])
        correct = bool(predictions.loc[claim_id, "correct"])

        # Try to load structured log
        structured_log: StructuredLog | None = None
        structured_log_path = claim_dir / "structured_log.json"

        if structured_log_path.exists():
            structured_log = load_structured_log_from_json(structured_log_path)

        # Fallback to text parsing if structured log not available
        if not structured_log:
            print(f"Using text log parsing for claim {claim_id}")
            warnings.warn(
                f"Falling back to text log parsing for claim {claim_id}. "
                "Consider regenerating structured logs for better reliability."
            )
            structured_log = load_structured_log_from_text(
                claim_dir=claim_dir,
                claim_id=str(claim_id),
                claim_text=claim_text,
                dataset=dataset,
                predicted_label=predicted_label,
            )

        # Skip if we couldn't load any log
        if not structured_log:
            warnings.warn(f"Could not load any log for claim {claim_id}, skipping this claim.")
            continue

        # Create ClaimData
        claim_data = ClaimData(
            log=structured_log,
            predicted_label=predicted_label,
            ground_truth_label=ground_truth_label,
            correct=correct,
        )
        claims.append(claim_data)

    return ExperimentData(
        experiment_dir=experiment_dir,
        claims=claims,
    )
