"""
JSON Schema definition for structured logging in DEFAME.

This schema defines the structure of structured log files that capture
all information about claim processing, including:
- Claim metadata
- Iterative evidence retrieval (tools, queries, results)
- Elaboration and reasoning
- Judgments and final verdicts
"""

from typing import Any

from pydantic import BaseModel, Field


class ClaimMetadata(BaseModel):
    """Metadata about the claim being verified."""
    id: str
    text: str
    dataset: str
    speaker: str | None = None
    date: str | None = None
    image_path: str | None = None
    source: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ResultInfo(BaseModel):
    """Information about a single evidence result."""
    url: str
    timestamp: str
    marked_useful: bool = False
    title: str | None = None
    snippet: str | None = None
    content: str | None = None
    summary: str | None = None
    platform: str | None = None
    error: str | None = None


class EvidenceRetrievalAction(BaseModel):
    """Information about a single evidence retrieval action."""
    action_type: str  # search, geolocate, object_detect, etc.
    tool: str
    timestamp: str
    results: list[ResultInfo] = Field(default_factory=list)
    total_results: int = 0
    unique_results: int = 0
    errors: list[str] = Field(default_factory=list)
    query: str | None = None
    platform: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    execution_time_seconds: float | None = None


class PlanningInfo(BaseModel):
    """Information about the planning phase."""
    timestamp: str
    plan_text: str
    actions_planned: list[str] = Field(default_factory=list)
    reasoning: str | None = None


class ElaborationInfo(BaseModel):
    """Information about the elaboration phase."""
    timestamp: str
    analysis_text: str
    extracted_facts: list[str] = Field(default_factory=list)
    reasoning: str | None = None


class JudgmentInfo(BaseModel):
    """Information about the judgment phase."""
    timestamp: str
    reasoning: str
    verdict: str  # NEI, SUPPORT, REFUTE, etc.
    confidence: str | None = None


class IterationInfo(BaseModel):
    """Information about a single iteration of the verification process."""
    iteration_number: int
    timestamp_start: str
    timestamp_end: str = ""
    planning: PlanningInfo = Field(default_factory=lambda: PlanningInfo(timestamp="", plan_text=""))
    evidence_retrieval: list[EvidenceRetrievalAction] = Field(default_factory=list)
    elaboration: ElaborationInfo = Field(default_factory=lambda: ElaborationInfo(timestamp="", analysis_text=""))
    judgment: JudgmentInfo = Field(default_factory=lambda: JudgmentInfo(timestamp="", reasoning="", verdict=""))


class FinalVerdict(BaseModel):
    """Final verdict information."""
    label: str
    justification: str
    timestamp: str
    confidence: str | None = None


class Statistics(BaseModel):
    """Statistics about the verification process."""
    total_iterations: int = 0
    total_searches: int = 0
    total_results: int = 0
    unique_sources: int = 0
    execution_time_seconds: float = 0.0
    model_calls: int = 0
    tokens_used: int | None = None


class StructuredLog(BaseModel):
    """Complete structured log for a claim verification."""
    version: str
    timestamp_start: str
    timestamp_end: str = ""
    claim: ClaimMetadata = Field(default_factory=lambda: ClaimMetadata(id="", text="", dataset=""))
    iterations: list[IterationInfo] = Field(default_factory=list)
    final_verdict: FinalVerdict = Field(default_factory=lambda: FinalVerdict(label="", justification="", timestamp=""))
    statistics: Statistics = Field(default_factory=Statistics)

    model_config = {
        "json_schema_extra": {
            "description": "Structured log for DEFAME claim verification"
        }
    }


# Schema version
SCHEMA_VERSION = "1.0.0"
