"""
Multi-experiment comparison logic for report generation.
"""

from collections import defaultdict
from typing import Any
import statistics

from defame.analysis.report.report_loader import ExperimentMetadata


def group_by_benchmark(experiments: list[ExperimentMetadata]) -> dict[str, list[ExperimentMetadata]]:
    """
    Group experiments by benchmark name.

    Args:
        experiments: List of experiment metadata

    Returns:
        Dictionary mapping benchmark name to list of experiments
    """
    groups = defaultdict(list)
    for exp in experiments:
        groups[exp.benchmark_name].append(exp)
    return dict(groups)


def group_by_model(experiments: list[ExperimentMetadata]) -> dict[str, list[ExperimentMetadata]]:
    """
    Group experiments by LLM model.

    Args:
        experiments: List of experiment metadata

    Returns:
        Dictionary mapping model name to list of experiments
    """
    groups = defaultdict(list)
    for exp in experiments:
        groups[exp.llm].append(exp)
    return dict(groups)


def get_metric_value(metadata: ExperimentMetadata, metric_path: str) -> Any:
    """
    Extract a metric value from experiment metadata using a path notation.

    Args:
        metadata: Experiment metadata
        metric_path: Path to the metric (e.g., "Predictions.Accuracy" or "search_coverage.metrics.avg_num_sources")

    Returns:
        The metric value, or None if not found
    """
    parts = metric_path.split(".")

    # Check if it's a performance metric or analysis result
    if parts[0] in ["Predictions", "Model", "Tools"]:
        # Performance metrics from results.json
        current = metadata.performance_metrics
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current
    else:
        # Analysis results - stored as dict
        current = metadata.analysis_results
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current


def compare_metric_across_experiments(
    experiments: list[ExperimentMetadata],
    metric_path: str
) -> dict[str, Any]:
    """
    Compare a specific metric across experiments.

    Args:
        experiments: List of experiment metadata
        metric_path: Path to the metric to compare

    Returns:
        Dictionary with comparison statistics
    """
    values = []
    exp_values = []

    for exp in experiments:
        value = get_metric_value(exp, metric_path)
        if value is not None and isinstance(value, (int, float)):
            # Convert to float to ensure consistent type
            float_value = float(value)
            # Skip NaN and infinity values
            if not (float('inf') == abs(float_value) or float_value != float_value):
                values.append(float_value)
                exp_values.append((exp, float_value))

    if not values:
        return {
            "metric_path": metric_path,
            "available": False,
            "values": []
        }

    result = {
        "metric_path": metric_path,
        "available": True,
        "values": exp_values,
        "count": len(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }

    if len(values) >= 2:
        try:
            result["stdev"] = statistics.stdev(values)
        except (ValueError, AttributeError, TypeError) as e:
            # Fallback to manual stdev calculation if statistics.stdev fails
            mean_val = result["mean"]
            variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
            result["stdev"] = variance ** 0.5
    else:
        result["stdev"] = 0.0

    return result


def identify_patterns(experiments: list[ExperimentMetadata]) -> dict[str, list[str]]:
    """
    Identify benchmark-specific vs model-specific vs general patterns.

    Args:
        experiments: List of experiment metadata

    Returns:
        Dictionary with pattern categories and insights
    """
    patterns = {
        "benchmark_specific": [],
        "model_specific": [],
        "general": []
    }

    # Group by benchmark and model
    by_benchmark = group_by_benchmark(experiments)
    by_model = group_by_model(experiments)

    # Benchmark-specific patterns
    if len(by_benchmark) > 1:
        for benchmark, exps in by_benchmark.items():
            if len(exps) > 1:
                # Check if this benchmark shows distinct patterns
                patterns["benchmark_specific"].append(
                    f"{len(exps)} experiments on {benchmark}"
                )

    # Model-specific patterns
    if len(by_model) > 1:
        for model, exps in by_model.items():
            if len(exps) > 1:
                patterns["model_specific"].append(
                    f"{len(exps)} experiments with {model}"
                )

    # General patterns
    if len(experiments) > 1:
        patterns["general"].append(
            f"Total of {len(experiments)} experiments analyzed"
        )

    return patterns


def calculate_deltas(
    experiments: list[ExperimentMetadata],
    metric_path: str,
    baseline_index: int = 0
) -> list[tuple[ExperimentMetadata, float, float]]:
    """
    Calculate deltas relative to a baseline experiment.

    Args:
        experiments: List of experiment metadata
        metric_path: Path to the metric to compare
        baseline_index: Index of the baseline experiment (default: 0)

    Returns:
        List of tuples (experiment, value, delta_from_baseline)
    """
    if not experiments or baseline_index >= len(experiments):
        return []

    baseline_value = get_metric_value(experiments[baseline_index], metric_path)
    if baseline_value is None or not isinstance(baseline_value, (int, float)):
        return []

    deltas = []
    for exp in experiments:
        value = get_metric_value(exp, metric_path)
        if value is not None and isinstance(value, (int, float)):
            delta = value - baseline_value
            deltas.append((exp, value, delta))

    return deltas


def find_best_experiment(
    experiments: list[ExperimentMetadata],
    metric_path: str,
    maximize: bool = True
) -> tuple[ExperimentMetadata, Any] | None:
    """
    Find the experiment with the best value for a given metric.

    Args:
        experiments: List of experiment metadata
        metric_path: Path to the metric
        maximize: Whether to maximize (True) or minimize (False) the metric

    Returns:
        Tuple of (best_experiment, best_value) or None if metric not found
    """
    best_exp = None
    best_value = None

    for exp in experiments:
        value = get_metric_value(exp, metric_path)
        if value is not None and isinstance(value, (int, float)):
            if best_value is None:
                best_exp = exp
                best_value = value
            elif (maximize and value > best_value) or (not maximize and value < best_value):
                best_exp = exp
                best_value = value

    if best_exp is None:
        return None

    return (best_exp, best_value)


def get_common_analyzers(experiments: list[ExperimentMetadata]) -> list[str]:
    """
    Get the list of analyzers that are present in all experiments.

    Args:
        experiments: List of experiment metadata

    Returns:
        List of analyzer names present in all experiments
    """
    if not experiments:
        return []

    # Get analyzers from the first experiment (now a dict)
    first_result = experiments[0].analysis_results
    analyzer_names = [key for key, value in first_result.items() if value is not None]

    # Filter to only those present in all experiments
    common_analyzers = []
    for analyzer in analyzer_names:
        if all(
            get_metric_value(exp, analyzer) is not None
            for exp in experiments
        ):
            common_analyzers.append(analyzer)

    return common_analyzers


def flatten_metrics(metrics_dict: dict, prefix: str = "") -> dict[str, Any]:
    """
    Flatten a nested metrics dictionary.

    Args:
        metrics_dict: Dictionary that may contain nested dictionaries
        prefix: Prefix for keys (used in recursion)

    Returns:
        Flattened dictionary with dot-notation keys
    """
    flattened = {}

    for key, value in metrics_dict.items():
        new_key = f"{prefix}.{key}" if prefix else key

        # If value is a dict, recursively flatten it
        if isinstance(value, dict):
            flattened.update(flatten_metrics(value, prefix=new_key))
        else:
            # It's a leaf value, add it
            flattened[new_key] = value

    return flattened


def compare_analyzer_metrics(
    experiments: list[ExperimentMetadata],
    analyzer_name: str
) -> dict[str, Any]:
    """
    Compare all metrics from a specific analyzer across experiments.

    Handles both flat and nested metric structures.

    Args:
        experiments: List of experiment metadata
        analyzer_name: Name of the analyzer (e.g., "search_coverage")

    Returns:
        Dictionary with metric comparisons
    """
    comparisons = {}

    # Get the metrics structure from the first experiment
    first_metrics = get_metric_value(experiments[0], f"{analyzer_name}.metrics")
    if first_metrics is None or not isinstance(first_metrics, dict):
        return comparisons

    # Flatten metrics in case they're nested (like reasoning analyzer)
    flattened_metrics = flatten_metrics(first_metrics)

    # Compare each metric
    for metric_name in flattened_metrics.keys():
        metric_path = f"{analyzer_name}.metrics.{metric_name}"
        comparison = compare_metric_across_experiments(experiments, metric_path)
        if comparison["available"]:
            comparisons[metric_name] = comparison

    return comparisons
