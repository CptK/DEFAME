"""
Configuration for report generation.
"""

from pydantic import BaseModel


class ReportConfig(BaseModel):
    """Configuration options for report generation."""

    include_extracted_data: bool = False
    """Whether to include raw extracted data from analyzers in the report."""

    include_all_metrics: bool = True
    """Whether to include all metrics or just summary metrics."""

    statistical_tests: bool = True
    """Whether to run statistical significance tests for comparisons."""

    grouping_priority: list[str] = ["benchmark", "model"]
    """Priority order for grouping experiments (e.g., ['benchmark', 'model'])."""

    show_experiment_paths: bool = True
    """Whether to show full experiment directory paths in the report."""

    precision: int = 3
    """Number of decimal places for displaying metrics."""
