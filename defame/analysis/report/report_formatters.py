"""
Markdown formatting utilities for report generation.
"""

from typing import Any


def format_header(title: str, level: int = 1) -> str:
    """
    Format a markdown header.

    Args:
        title: Header text
        level: Header level (1-6)

    Returns:
        Formatted markdown header
    """
    return f"{'#' * level} {title}\n"


def format_table(headers: list[str], rows: list[list[Any]], alignments: list[str] | None = None) -> str:
    """
    Format a markdown table.

    Args:
        headers: List of column headers
        rows: List of rows, where each row is a list of cell values
        alignments: Optional list of alignments ('left', 'center', 'right') for each column

    Returns:
        Formatted markdown table
    """
    if not headers or not rows:
        return ""

    # Convert all cells to strings
    str_headers = [str(h) for h in headers]
    str_rows = [[str(cell) for cell in row] for row in rows]

    # Calculate column widths
    col_widths = [len(h) for h in str_headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # Format header row
    header_row = "| " + " | ".join(str_headers) + " |"

    # Format separator row
    if alignments is None:
        alignments = ["left"] * len(headers)

    sep_parts = []
    for i, alignment in enumerate(alignments):
        width = col_widths[i]
        if alignment == "center":
            sep_parts.append(":" + "-" * (width - 2) + ":")
        elif alignment == "right":
            sep_parts.append("-" * (width - 1) + ":")
        else:  # left
            sep_parts.append("-" * width)

    separator_row = "| " + " | ".join(sep_parts) + " |"

    # Format data rows
    data_rows = []
    for row in str_rows:
        data_rows.append("| " + " | ".join(row) + " |")

    return "\n".join([header_row, separator_row] + data_rows) + "\n"


def format_percentage(value: float, precision: int = 1) -> str:
    """
    Format a float as a percentage.

    Args:
        value: Value to format (0-1 range)
        precision: Number of decimal places

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{precision}f}%"


def format_number(value: float, precision: int = 3) -> str:
    """
    Format a number with specified precision.

    Args:
        value: Value to format
        precision: Number of decimal places

    Returns:
        Formatted number string
    """
    return f"{value:.{precision}f}"


def format_delta(value: float, precision: int = 1, as_percentage: bool = True) -> str:
    """
    Format a delta value with +/- sign and optional arrow indicator.

    Args:
        value: Delta value
        precision: Number of decimal places
        as_percentage: Whether to format as percentage

    Returns:
        Formatted delta string with indicator
    """
    if as_percentage:
        formatted = f"{value * 100:+.{precision}f}%"
    else:
        formatted = f"{value:+.{precision}f}"

    # Add arrow indicator
    if value > 0:
        return f"↑ {formatted}"
    elif value < 0:
        return f"↓ {formatted}"
    else:
        return f"→ {formatted}"


def format_correlation(r: float, p: float, precision: int = 2) -> str:
    """
    Format a correlation coefficient and p-value.

    Args:
        r: Correlation coefficient
        p: P-value
        precision: Number of decimal places

    Returns:
        Formatted correlation string
    """
    significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    return f"r={r:.{precision}f}, p={p:.3f}{significance}"


def format_timestamp(timestamp: str | None) -> str:
    """
    Format a timestamp string for display.

    Args:
        timestamp: Timestamp string (e.g., "2025-05-27_06-47")

    Returns:
        Formatted timestamp or "N/A" if None
    """
    if timestamp is None:
        return "N/A"

    # Try to parse and reformat
    try:
        # Handle format like "2025-05-27_06-47"
        if "_" in timestamp:
            date_part, time_part = timestamp.split("_")
            time_part = time_part.replace("-", ":")
            return f"{date_part} {time_part}"
        return timestamp
    except:
        return timestamp


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_list(items: list[str], numbered: bool = False) -> str:
    """
    Format a list in markdown.

    Args:
        items: List of items
        numbered: Whether to use numbered list

    Returns:
        Formatted markdown list
    """
    if numbered:
        return "\n".join([f"{i+1}. {item}" for i, item in enumerate(items)]) + "\n"
    else:
        return "\n".join([f"- {item}" for item in items]) + "\n"


def format_code_block(content: str, language: str = "") -> str:
    """
    Format a code block in markdown.

    Args:
        content: Code content
        language: Programming language for syntax highlighting

    Returns:
        Formatted code block
    """
    return f"```{language}\n{content}\n```\n"


def format_bold(text: str) -> str:
    """Format text as bold."""
    return f"**{text}**"


def format_italic(text: str) -> str:
    """Format text as italic."""
    return f"*{text}*"


def format_metric_value(value: Any, metric_name: str = "", precision: int = 3) -> str:
    """
    Intelligently format a metric value based on its type and name.

    Args:
        value: The metric value to format
        metric_name: Name of the metric (used to infer formatting)
        precision: Number of decimal places for floats

    Returns:
        Formatted metric value
    """
    if value is None:
        return "N/A"

    # Check if this is a percentage-like metric
    percentage_keywords = ["accuracy", "precision", "recall", "f1"]
    is_percentage = any(keyword in metric_name.lower() for keyword in percentage_keywords)

    if isinstance(value, bool):
        return "✓" if value else "✗"
    elif isinstance(value, float):
        if is_percentage and 0 <= value <= 1:
            return format_percentage(value, precision=precision)
        else:
            return format_number(value, precision=precision)
    elif isinstance(value, int):
        return str(value)
    elif isinstance(value, str):
        return value
    elif isinstance(value, (list, tuple)):
        # For correlation tuples like (r, p)
        if len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
            return format_correlation(value[0], value[1])
        return str(value)
    else:
        return str(value)
