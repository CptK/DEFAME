"""
Base analyzer class for all DEFAME log analyzers.

This module defines the abstract interface that all analyzers must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
from pydantic import BaseModel

from defame.analysis.data_models import ExperimentData
from defame.analysis.analyzer_config import AnalyzerConfig


ExtractedDataType = TypeVar('ExtractedDataType', bound=BaseModel)
MetricsType = TypeVar('MetricsType', bound=BaseModel)
InsightsType = TypeVar('InsightsType', bound=BaseModel)


class AnalysisResult(BaseModel, Generic[ExtractedDataType, MetricsType, InsightsType]):
    analyzer_name: str
    extracted_data: list[ExtractedDataType]
    metrics: MetricsType
    insights: InsightsType


class BaseAnalyzer(ABC, Generic[ExtractedDataType, MetricsType, InsightsType]):
    """
    Abstract base class for all analyzers.

    All analyzers should inherit from this class and implement:
    - extract_data: Extract relevant data from parsed iterations
    - compute_metrics: Compute metrics from extracted data
    - generate_insights: Generate high-level insights from metrics

    The analyze() method orchestrates these three steps.
    """

    def __init__(self, config: AnalyzerConfig):
        """
        Initialize the analyzer.

        Args:
            config: Configuration for the analyzer
        """
        self.config = config

    @abstractmethod
    def extract_data(self, experiment_data: ExperimentData) -> list[ExtractedDataType]:
        """
        Extract relevant data from experiment.

        Args:
            experiment_data: ExperimentData object containing claim data

        Returns:
            List of extracted data
        """
        pass

    @abstractmethod
    def compute_metrics(self, extracted_data: list[ExtractedDataType]) -> MetricsType:
        """
        Compute metrics from extracted data.

        Args:
            extracted_data: Data extracted from iterations

        Returns:
            Computed metrics
        """
        pass

    @abstractmethod
    def generate_insights(self, metrics: MetricsType) -> InsightsType:
        """
        Generate high-level insights from metrics.

        Args:
            metrics: Computed metrics

        Returns:
            Extracted insights
        """
        pass

    def analyze(self, experiment_data: ExperimentData) -> AnalysisResult:
        """
        Main analysis pipeline.

        Executes the three-stage analysis process:
        1. Extract data from experiment
        2. Compute metrics from extracted data
        3. Generate insights from metrics

        Args:
            experiment_data: ExperimentData object containing claim data

        Returns:
            AnalysisResult containing metrics, insights, and metadata
        """
        extracted = self.extract_data(experiment_data)
        metrics = self.compute_metrics(extracted)
        insights = self.generate_insights(metrics)

        return AnalysisResult(
            analyzer_name=self.__class__.__name__,
            extracted_data=extracted,
            metrics=metrics,
            insights=insights
        )
