"""
Library and script to parse DEFAME fact-check log files by iteration.

Core function:
    parse_log_iterations(log_path: str | Path) -> List[str]
        Returns a list of strings, where each string is the content of one iteration.

The primary marker for iteration boundaries is the line:
"Not enough information yet. Continuing fact-check..."

This line appears at the start of each new iteration (except the first one).
"""

from pathlib import Path


def parse_log_iterations(log_path: str | Path) -> list[str]:
    """
    Parse a log file and return iterations as a list of strings.

    This is the core function for processing log files. Each iteration is returned
    as a single string containing all log content for that iteration.

    Args:
        log_path: Path to the log.txt file (string or Path object)

    Returns:
        List of strings, where each string contains the full log content for one iteration

    Raises:
        FileNotFoundError: If the log file doesn't exist

    Example:
        >>> iterations = parse_log_iterations('/path/to/fact-checks/123/log.txt')
        >>> print(f"Found {len(iterations)} iterations")
        >>> print(f"First iteration length: {len(iterations[0])} characters")
        >>> # Process each iteration
        >>> for i, iteration_content in enumerate(iterations, start=1):
        ...     print(f"Iteration {i}: {iteration_content[:100]}...")
    """
    log_path = Path(log_path)

    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    # Read the entire log file
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Marker that indicates the start of a new iteration (iterations 2+)
    iteration_marker = "Not enough information yet. Continuing fact-check..."

    # Split by marker
    parts = content.split(iteration_marker)

    # First part is iteration 1
    # Subsequent parts need the marker prepended (since it was removed by split)
    iterations = [parts[0]]
    for part in parts[1:]:
        iterations.append(iteration_marker + part)

    # Remove empty iterations (shouldn't happen, but just in case)
    iterations = [it.strip() for it in iterations if it.strip()]

    return iterations[1:]


def write_iterations_to_files(log_path: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    """
    Parse a log file and write each iteration to a separate file.

    Args:
        log_path: Path to the log.txt file
        output_dir: Directory to write split files (defaults to same directory as log file)

    Returns:
        List of paths to the created iteration log files

    Example:
        >>> paths = write_iterations_to_files('/path/to/log.txt', '/output/dir')
        >>> print(f"Created {len(paths)} files")
    """
    log_path = Path(log_path)

    # Use same directory as log file if no output directory specified
    if output_dir is None:
        output_dir = log_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Get iterations
    iterations = parse_log_iterations(log_path)

    # Write each iteration to a separate file
    output_paths = []
    for i, iteration_content in enumerate(iterations, start=1):
        output_path = output_dir / f"log_iter{i}.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(iteration_content)
        output_paths.append(output_path)
        print(f"Iteration {i}: {len(iteration_content)} chars -> {output_path}")

    return output_paths


def get_iteration_stats(log_path: str | Path) -> dict:
    """
    Get statistics about iterations in a log file.

    Args:
        log_path: Path to the log.txt file

    Returns:
        Dictionary with iteration statistics:
            - num_iterations: Number of iterations
            - iteration_lengths: List of character counts for each iteration
            - total_chars: Total characters in file

    Example:
        >>> stats = get_iteration_stats('/path/to/log.txt')
        >>> print(f"Found {stats['num_iterations']} iterations")
        >>> for i, length in enumerate(stats['iteration_lengths'], start=1):
        ...     print(f"Iteration {i}: {length} characters")
    """
    iterations = parse_log_iterations(log_path)

    return {
        'num_iterations': len(iterations),
        'iteration_lengths': [len(it) for it in iterations],
        'total_chars': sum(len(it) for it in iterations)
    }


def process_experiment_logs(experiment_dir: str | Path, fact_check_ids: list[str] | None = None) -> dict:
    """
    Parse log files for multiple fact-checks in an experiment directory.

    Args:
        experiment_dir: Path to the experiment output directory
        fact_check_ids: List of specific fact-check IDs to process (None = all)

    Returns:
        Dictionary mapping fact-check ID to list of iteration strings

    Example:
        >>> results = process_experiment_logs('/path/to/experiment')
        >>> for fc_id, iterations in results.items():
        ...     print(f"Fact-check {fc_id}: {len(iterations)} iterations")
    """
    experiment_dir = Path(experiment_dir)
    fact_checks_dir = experiment_dir / "fact-checks"

    if not fact_checks_dir.exists():
        raise FileNotFoundError(f"Fact-checks directory not found: {fact_checks_dir}")

    results = {}

    # Get list of fact-check directories
    if fact_check_ids is None:
        fc_dirs = [d for d in fact_checks_dir.iterdir() if d.is_dir()]
    else:
        fc_dirs = [fact_checks_dir / str(fc_id) for fc_id in fact_check_ids]

    # Process each fact-check
    for fc_dir in sorted(fc_dirs):
        if not fc_dir.is_dir():
            continue

        log_path = fc_dir / "log.txt"
        if not log_path.exists():
            continue

        fc_id = fc_dir.name

        try:
            iterations = parse_log_iterations(log_path)
            results[fc_id] = iterations
        except Exception as e:
            print(f"Warning: Error processing fact-check {fc_id}: {e}")

    return results


def write_experiment_logs(experiment_dir: str | Path, fact_check_ids: list[str] | None = None) -> dict:
    """
    Split log files for multiple fact-checks in an experiment directory and write to files.

    Args:
        experiment_dir: Path to the experiment output directory
        fact_check_ids: List of specific fact-check IDs to process (None = all)

    Returns:
        Dictionary mapping fact-check ID to list of iteration file paths
    """
    experiment_dir = Path(experiment_dir)
    fact_checks_dir = experiment_dir / "fact-checks"

    if not fact_checks_dir.exists():
        raise FileNotFoundError(f"Fact-checks directory not found: {fact_checks_dir}")

    results = {}

    # Get list of fact-check directories
    if fact_check_ids is None:
        fc_dirs = [d for d in fact_checks_dir.iterdir() if d.is_dir()]
    else:
        fc_dirs = [fact_checks_dir / str(fc_id) for fc_id in fact_check_ids]

    # Process each fact-check
    for fc_dir in sorted(fc_dirs):
        if not fc_dir.is_dir():
            continue

        log_path = fc_dir / "log.txt"
        if not log_path.exists():
            print(f"Warning: No log.txt found in {fc_dir}")
            continue

        fc_id = fc_dir.name
        print(f"\nProcessing fact-check {fc_id}...")

        try:
            iteration_paths = write_iterations_to_files(log_path)
            results[fc_id] = iteration_paths
            print(f"✓ Created {len(iteration_paths)} iteration files for fact-check {fc_id}")
        except Exception as e:
            print(f"✗ Error processing fact-check {fc_id}: {e}")

    return results


if __name__ == "__main__":
    BASE_DIR = "/mnt/vast/workspaces/PI_Rohrbach/mk79honu/DEFAME/out/verite/summary/dynamic/llama4_scout/2025-06-03_19-32 defame"
    results = process_experiment_logs(BASE_DIR)
    for i, (fc_id, iterations) in enumerate(results.items(), start=1):
        print(f"Fact-check {i} ({fc_id}): {len(iterations)} iterations")

        if i >= 5:
            break
