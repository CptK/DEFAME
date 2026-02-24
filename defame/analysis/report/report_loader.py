"""
Load experiment metadata from directories for report generation.
"""

import json
import re
import yaml
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field


class ExperimentMetadata(BaseModel):
    """Metadata for a single experiment."""

    experiment_dir: Path
    """Path to the experiment directory."""

    experiment_name: str | None = None
    """Name of the experiment (from config.yaml)."""

    benchmark_name: str
    """Name of the benchmark (e.g., 'averitec', 'mocheg')."""

    llm: str
    """Name of the LLM model used."""

    timestamp: str | None = None
    """Timestamp extracted from directory name."""

    performance_metrics: dict[str, Any]
    """Performance metrics from results.json (accuracy, F1, etc.)."""

    analysis_results: dict[str, Any]
    """Analysis results from analysis_results.json (stored as dict to avoid Pydantic Generic issues)."""

    config: dict[str, Any] = Field(default_factory=dict)
    """Full configuration from config.yaml."""


def validate_experiment_dir(experiment_dir: Path) -> tuple[bool, list[str]]:
    """
    Check if directory has required files for report generation.

    Args:
        experiment_dir: Path to the experiment directory

    Returns:
        Tuple of (is_valid, missing_files)
    """
    required_files = {
        "analysis_results.json": experiment_dir / "analysis_results.json",
        "config.yaml": experiment_dir / "config.yaml",
        "results.json": experiment_dir / "results.json",
    }

    missing_files = []
    for name, path in required_files.items():
        if not path.exists():
            # Try alternative for results (yaml instead of json)
            if name == "results.json":
                alt_path = experiment_dir / "results.yaml"
                if not alt_path.exists():
                    missing_files.append(name)
            else:
                missing_files.append(name)

    is_valid = len(missing_files) == 0
    return is_valid, missing_files


def extract_timestamp_from_path(path: Path) -> str | None:
    """
    Extract timestamp from directory path.

    Args:
        path: Path to the experiment directory

    Returns:
        Timestamp string if found, None otherwise
    """
    # Look for patterns like "2025-05-27_06-47"
    timestamp_pattern = r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}'
    match = re.search(timestamp_pattern, str(path))
    if match:
        return match.group(0)
    return None


def load_experiment_metadata(experiment_dir: str | Path) -> ExperimentMetadata:
    """
    Load all metadata from an experiment directory.

    Args:
        experiment_dir: Path to the experiment directory

    Returns:
        ExperimentMetadata object with all loaded data

    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If required fields are missing from config
    """
    experiment_dir = Path(experiment_dir)

    # Validate directory
    is_valid, missing_files = validate_experiment_dir(experiment_dir)
    if not is_valid:
        raise FileNotFoundError(
            f"Experiment directory {experiment_dir} is missing required files: {missing_files}"
        )

    # Load config.yaml
    config_path = experiment_dir / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract required fields from config
    benchmark_name = config.get("benchmark_name")
    if not benchmark_name:
        raise ValueError(f"config.yaml in {experiment_dir} is missing 'benchmark_name'")

    llm = config.get("llm")
    if not llm:
        raise ValueError(f"config.yaml in {experiment_dir} is missing 'llm'")

    experiment_name = config.get("experiment_name")

    # Load performance metrics (try json first, then yaml)
    results_json_path = experiment_dir / "results.json"
    results_yaml_path = experiment_dir / "results.yaml"

    if results_json_path.exists():
        with open(results_json_path, "r") as f:
            performance_metrics = json.load(f)
    elif results_yaml_path.exists():
        with open(results_yaml_path, "r") as f:
            performance_metrics = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"No results file found in {experiment_dir}")

    # Load analysis results as dict (avoid Pydantic Generic issues)
    analysis_results_path = experiment_dir / "analysis_results.json"
    with open(analysis_results_path, "r") as f:
        analysis_results = json.load(f)

    # Extract timestamp from directory path
    timestamp = extract_timestamp_from_path(experiment_dir)

    return ExperimentMetadata(
        experiment_dir=experiment_dir,
        experiment_name=experiment_name,
        benchmark_name=benchmark_name,
        llm=llm,
        timestamp=timestamp,
        performance_metrics=performance_metrics,
        analysis_results=analysis_results,
        config=config,
    )


def load_multiple_experiments(
    experiment_dirs: list[str | Path]
) -> list[ExperimentMetadata]:
    """
    Load metadata from multiple experiments, with validation.

    Args:
        experiment_dirs: List of paths to experiment directories

    Returns:
        List of ExperimentMetadata objects

    Raises:
        ValueError: If no valid experiments are found
    """
    experiments = []
    errors = []

    for exp_dir in experiment_dirs:
        try:
            metadata = load_experiment_metadata(exp_dir)
            experiments.append(metadata)
        except Exception as e:
            errors.append((exp_dir, str(e)))

    if errors:
        print("Warning: Some experiments could not be loaded:")
        for exp_dir, error in errors:
            print(f"  {exp_dir}: {error}")

    if not experiments:
        raise ValueError("No valid experiments found. All directories failed to load.")

    return experiments
