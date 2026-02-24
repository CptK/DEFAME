"""
Main report generator for DEFAME analysis results.
"""

from datetime import datetime
from pathlib import Path

from defame.analysis.report.report_config import ReportConfig
from defame.analysis.report.report_loader import ExperimentMetadata, load_multiple_experiments
from defame.analysis.report.report_formatters import (
    format_header,
    format_table,
    format_percentage,
    format_number,
    format_timestamp,
    format_delta,
    format_metric_value,
    format_bold,
    format_list,
)
from defame.analysis.report.report_comparison import (
    get_common_analyzers,
    compare_analyzer_metrics,
    find_best_experiment,
    group_by_benchmark,
    group_by_model,
    get_metric_value,
)
from defame.analysis.report.report_pdf import markdown_to_pdf


class ReportGenerator:
    """Generate markdown reports from DEFAME analysis results."""

    def __init__(
        self,
        experiments: list[ExperimentMetadata],
        config: ReportConfig | None = None
    ):
        """
        Initialize the report generator.

        Args:
            experiments: List of experiment metadata
            config: Report configuration (uses defaults if None)
        """
        self.experiments = experiments
        self.config = config or ReportConfig()
        self.is_comparison = len(experiments) > 1

    def generate(self) -> str:
        """
        Generate the complete markdown report.

        Returns:
            Markdown-formatted report as a string
        """
        sections = [
            self._generate_title(),
            self._generate_metadata_section(),
            self._generate_executive_summary(),
            self._generate_pattern_analysis(),
            self._generate_analyzer_sections(),
            self._generate_footer(),
        ]

        # Filter out empty sections
        sections = [s for s in sections if s.strip()]

        return "\n\n".join(sections)

    def _generate_title(self) -> str:
        """Generate the report title."""
        title = "DEFAME Analysis Report"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return format_header(title, level=1) + f"*Generated: {timestamp}*\n"

    def _generate_metadata_section(self) -> str:
        """Generate the experiments metadata table."""
        section = format_header("Experiments Analyzed", level=2)

        headers = ["ID", "Experiment", "Benchmark", "Model", "Timestamp"]
        if self.config.show_experiment_paths:
            headers.append("Directory")

        rows = []
        for i, exp in enumerate(self.experiments, 1):
            row = [
                str(i),
                exp.experiment_name or "N/A",
                exp.benchmark_name,
                exp.llm,
                format_timestamp(exp.timestamp),
            ]
            if self.config.show_experiment_paths:
                row.append(str(exp.experiment_dir))
            rows.append(row)

        section += format_table(headers, rows)
        return section

    def _generate_executive_summary(self) -> str:
        """Generate the executive summary with performance metrics."""
        section = format_header("Performance Summary", level=2)

        # Define metrics to extract
        metric_configs = [
            ("Predictions.Accuracy", "Accuracy", True),
            ("Predictions.Macro-Averaged F1-Score", "Macro F1", False),
            ("Predictions.Total", "Samples", False),
            ("Predictions.Correct", "Correct", False),
            ("Predictions.Wrong", "Wrong", False),
            ("Time per claim", "Time/Claim", False),
        ]

        # Build headers
        headers = ["Exp"]
        for _, label, _ in metric_configs:
            headers.append(label)

        # Build rows
        rows = []
        for i, exp in enumerate(self.experiments, 1):
            row = [str(i)]
            for metric_path, _, is_percentage in metric_configs:
                value = get_metric_value(exp, metric_path)
                if value is not None:
                    if is_percentage and isinstance(value, (int, float)):
                        row.append(format_percentage(value, precision=1))
                    elif isinstance(value, float):
                        row.append(format_number(value, precision=self.config.precision))
                    else:
                        row.append(str(value))
                else:
                    row.append("N/A")
            rows.append(row)

        section += format_table(headers, rows)

        # Add best performer if comparing multiple experiments
        if self.is_comparison:
            section += "\n" + format_bold("Best Performers:") + "\n\n"
            best_items = []

            accuracy_best = find_best_experiment(self.experiments, "Predictions.Accuracy", maximize=True)
            if accuracy_best:
                exp_idx = self.experiments.index(accuracy_best[0]) + 1
                best_items.append(f"Accuracy: Experiment {exp_idx} ({format_percentage(accuracy_best[1])})")

            f1_best = find_best_experiment(self.experiments, "Predictions.Macro-Averaged F1-Score", maximize=True)
            if f1_best:
                exp_idx = self.experiments.index(f1_best[0]) + 1
                best_items.append(f"Macro F1: Experiment {exp_idx} ({format_number(f1_best[1], precision=3)})")

            section += format_list(best_items)

        return section

    def _generate_pattern_analysis(self) -> str:
        """Generate pattern analysis section for multi-experiment comparisons."""
        if not self.is_comparison:
            return ""

        section = format_header("Pattern Analysis", level=2)

        # Group by benchmark
        by_benchmark = group_by_benchmark(self.experiments)
        if len(by_benchmark) > 1:
            section += format_header("By Benchmark", level=3)
            items = []
            for benchmark, exps in by_benchmark.items():
                items.append(f"{format_bold(benchmark)}: {len(exps)} experiment(s)")
            section += format_list(items)

        # Group by model
        by_model = group_by_model(self.experiments)
        if len(by_model) > 1:
            section += format_header("By Model", level=3)
            items = []
            for model, exps in by_model.items():
                items.append(f"{format_bold(model)}: {len(exps)} experiment(s)")
            section += format_list(items)

        return section

    def _generate_analyzer_sections(self) -> str:
        """Generate sections for each analyzer."""
        section = format_header("Detailed Analysis", level=2)

        # Get analyzers present in all experiments
        common_analyzers = get_common_analyzers(self.experiments)

        if not common_analyzers:
            section += "*No common analyzers found across all experiments.*\n"
            return section

        # Generate section for each analyzer
        for analyzer_name in common_analyzers:
            section += self._generate_analyzer_section(analyzer_name)

        return section

    def _generate_analyzer_section(self, analyzer_name: str) -> str:
        """
        Generate a section for a specific analyzer.

        Args:
            analyzer_name: Name of the analyzer (e.g., "search_coverage")

        Returns:
            Formatted section for the analyzer
        """
        # Create human-readable title
        title = analyzer_name.replace("_", " ").title()
        section = format_header(title, level=3)

        # Get metrics comparison
        metric_comparisons = compare_analyzer_metrics(self.experiments, analyzer_name)

        if not metric_comparisons:
            section += "*No metrics available for this analyzer.*\n"
            return section

        # Build metrics comparison table
        section += format_header("Metrics Comparison", level=4)

        headers = ["Metric"] + [f"Exp {i+1}" for i in range(len(self.experiments))]
        if self.is_comparison:
            headers.extend(["Mean", "Min", "Max"])

        rows = []
        for metric_name, comparison in metric_comparisons.items():
            # Format metric name
            formatted_name = metric_name.replace("_", " ").title()
            row = [formatted_name]

            # Add values for each experiment
            for exp, value in comparison["values"]:
                formatted_value = format_metric_value(value, metric_name, self.config.precision)
                row.append(formatted_value)

            # Add statistics for comparisons
            if self.is_comparison:
                row.append(format_number(comparison["mean"], precision=self.config.precision))
                row.append(format_number(comparison["min"], precision=self.config.precision))
                row.append(format_number(comparison["max"], precision=self.config.precision))

            rows.append(row)

        section += format_table(headers, rows)

        # Add insights section
        section += self._generate_analyzer_insights(analyzer_name)

        return section

    def _generate_analyzer_insights(self, analyzer_name: str) -> str:
        """
        Generate insights section for an analyzer.

        Args:
            analyzer_name: Name of the analyzer

        Returns:
            Formatted insights section
        """
        section = format_header("Insights", level=4)

        insights_found = False

        for i, exp in enumerate(self.experiments, 1):
            insights = get_metric_value(exp, f"{analyzer_name}.insights")
            if insights is None:
                continue

            insights_found = True

            if self.is_comparison:
                section += format_bold(f"Experiment {i}:") + "\n\n"

            # Handle different insights formats
            if isinstance(insights, dict):
                # Extract insights_summary if available
                if "insights_summary" in insights:
                    section += insights["insights_summary"] + "\n\n"
                else:
                    # Format all insights
                    for key, value in insights.items():
                        section += f"- {key}: {value}\n"
                    section += "\n"
            elif isinstance(insights, str):
                section += insights + "\n\n"
            else:
                section += str(insights) + "\n\n"

        if not insights_found:
            section += "*No insights available for this analyzer.*\n"

        return section

    def _generate_footer(self) -> str:
        """Generate the report footer."""
        footer = "---\n\n"
        footer += f"*Report generated by DEFAME Analysis Framework*\n"
        return footer


def generate_report(
    experiment_dirs: list[str | Path],
    output_path: str | Path | None = None,
    config: ReportConfig | None = None,
    generate_pdf: bool = True
) -> str:
    """
    Generate a markdown report from experiment directories.

    Args:
        experiment_dirs: List of paths to experiment directories
        output_path: Optional path to save the report (if None, only returns string)
        config: Report configuration (uses defaults if None)
        generate_pdf: Whether to also generate a PDF version (default: True)

    Returns:
        Markdown-formatted report as a string

    Raises:
        ValueError: If no valid experiments are found
    """
    # Load experiments
    experiments = load_multiple_experiments(experiment_dirs)

    # Generate report
    generator = ReportGenerator(experiments, config)
    report = generator.generate()

    # Save to file if path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save markdown
        with open(output_path, "w") as f:
            f.write(report)
        print(f"✓ Markdown report saved to: {output_path}")

        # Generate PDF if requested
        if generate_pdf:
            # Determine PDF output path (same name, .pdf extension)
            pdf_path = output_path.with_suffix('.pdf')

            result = markdown_to_pdf(report, pdf_path)
            if result and "PDF generated:" in result:
                print(f"✓ {result}")
            else:
                print(f"✗ {result}")

    return report
