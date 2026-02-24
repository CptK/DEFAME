from pydantic import BaseModel, Field
import json

from defame.analysis.loading import load_experiment_logs
from defame.analysis.analyzers import (
    SearchCoverageAnalyzer,
    QueryStrategyAnalyzer,
    SourceUtilizationAnalyzer,
    ReasoningAnalyzer,
    ConfidenceAnalyzer,
    IterationAnalyzer
)
from defame.analysis.analyzers.base_analyzer import AnalysisResult
from defame.analysis.analyzer_config import AnalyzerConfig


class AnalysisResults(BaseModel):
    search_coverage: AnalysisResult | None = Field(
        default=None, description="Results from the Search Coverage Analyzer"
    )
    query_strategy: AnalysisResult | None = Field(
        default=None, description="Results from the Query Strategy Analyzer"
    )
    source_utilization: AnalysisResult | None = Field(
        default=None, description="Results from the Source Utilization Analyzer"
    )
    reasoning: AnalysisResult | None = Field(
        default=None, description="Results from the Reasoning Analyzer"
    )
    confidence: AnalysisResult | None = Field(
        default=None, description="Results from the Confidence Analyzer"
    )
    iterations: AnalysisResult | None = Field(
        default=None, description="Results from the Iteration Analyzer"
    )



class AnalysisPipeline:
    def __init__(
        self,
        analyze_search_coverage: bool = True,
        analyze_query_strategy: bool = True,
        analyze_source_utilization: bool = True,
        analyze_reasoning: bool = True,
        analyze_confidence: bool = True,
        analyze_iterations: bool = True
    ):
        self.analyze_search_coverage = analyze_search_coverage
        self.analyze_query_strategy = analyze_query_strategy
        self.analyze_source_utilization = analyze_source_utilization
        self.analyze_reasoning = analyze_reasoning
        self.analyze_confidence = analyze_confidence
        self.analyze_iterations = analyze_iterations

    def analyze(
        self,
        experiment_dir: str,
        config: AnalyzerConfig,
        first_n: int | None = None,
        save: bool = True
    ) -> AnalysisResults:
        """Run the selected analyzers on the experiment logs.
        
        Args:
            experiment_dir: Directory containing the experiment logs
            config: AnalyzerConfig object with configuration for analyzers
            first_n: If specified, only analyze the first N claim logs
            save: Whether to save the results to a JSON file in the experiment directory
        
        Returns:
            AnalysisResults object containing results from the selected analyzers
        """
        experiment_log = load_experiment_logs(experiment_dir)
        if first_n is not None:
            experiment_log.claims = experiment_log.claims[:first_n]

        results = AnalysisResults()

        if self.analyze_search_coverage:
            search_coverage_analyzer = SearchCoverageAnalyzer(config=config)
            results.search_coverage = search_coverage_analyzer.analyze(experiment_log)

        if self.analyze_query_strategy:
            query_strategy_analyzer = QueryStrategyAnalyzer(config=config)
            results.query_strategy = query_strategy_analyzer.analyze(experiment_log)

        if self.analyze_source_utilization:
            source_utilization_analyzer = SourceUtilizationAnalyzer(config=config)
            results.source_utilization = source_utilization_analyzer.analyze(experiment_log)

        if self.analyze_reasoning:
            reasoning_analyzer = ReasoningAnalyzer(config=config)
            results.reasoning = reasoning_analyzer.analyze(experiment_log)

        if self.analyze_confidence:
            confidence_analyzer = ConfidenceAnalyzer(config=config)
            results.confidence = confidence_analyzer.analyze(experiment_log)

        if self.analyze_iterations:
            iteration_analyzer = IterationAnalyzer(config=config)
            results.iterations = iteration_analyzer.analyze(experiment_log)
        
        if save:
            self.save_results(results, experiment_dir)

        return results
    
    def save_results(self, results: AnalysisResults, output_dir: str, filename: str = "analysis_results.json"):
        """Save the analysis results to a JSON file.

        Args:
            results: AnalysisResults object to save
            output_dir: Directory to save the results in
            filename: Name of the JSON file
        """
        with open(f"{output_dir}/{filename}", "w") as f:
            json.dump(results.model_dump(), f, indent=4)
            print(f"Saved analysis results to {output_dir}/{filename}")
