"""
Data models for DEFAME analysis.

This module defines the data structures used for analyzing fact-checking experiments.
These models wrap structured logs with evaluation metadata.
"""

from pathlib import Path
from pydantic import BaseModel

from defame.common.structured_log_schema import StructuredLog


class ClaimData(BaseModel):
    """
    Data for a single claim to be analyzed.

    Combines the structured log (containing all fact-checking data) with
    evaluation metadata (predicted vs ground truth labels).
    """

    log: StructuredLog
    predicted_label: str
    ground_truth_label: str
    correct: bool

    @property
    def claim_id(self) -> str:
        """Convenience property to access claim ID."""
        return self.log.claim.id

    @property
    def claim_text(self) -> str:
        """Convenience property to access claim text."""
        return self.log.claim.text


class ExperimentData(BaseModel):
    """
    Data for an entire experiment to be analyzed.

    Contains all claims in the experiment along with experiment metadata.
    """

    experiment_dir: Path
    claims: list[ClaimData]

    @property
    def num_claims(self) -> int:
        """Total number of claims in the experiment."""
        return len(self.claims)

    @property
    def num_correct(self) -> int:
        """Number of correctly predicted claims."""
        return sum(1 for claim in self.claims if claim.correct)

    @property
    def accuracy(self) -> float:
        """Accuracy of predictions."""
        if self.num_claims == 0:
            return 0.0
        return self.num_correct / self.num_claims
