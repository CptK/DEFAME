"""
StructuredLogger for DEFAME claim verification.

This module provides a logger that writes structured JSON logs capturing
all information about the claim verification process.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from defame.common.structured_log_schema import (
    StructuredLog,
    ClaimMetadata,
    IterationInfo,
    PlanningInfo,
    ElaborationInfo,
    JudgmentInfo,
    EvidenceRetrievalAction,
    ResultInfo,
    FinalVerdict,
    Statistics,
    SCHEMA_VERSION,
)


class StructuredLogger:
    """
    Thread-safe structured logger that maintains an in-memory Pydantic model
    and writes incrementally to disk after each iteration.
    """

    def __init__(self, output_path: Path):
        """
        Initialize the structured logger.

        Args:
            output_path: Path where the structured log JSON file will be written.
        """
        self.output_path = Path(output_path)
        self.lock = Lock()

        # Initialize the log structure as a Pydantic model
        self.log = StructuredLog(
            version=SCHEMA_VERSION,
            timestamp_start=self._get_timestamp(),
        )

        # Track timing
        self.start_time = time.time()
        self.iteration_start_time = None

        # Track statistics
        self.stats = {
            "total_searches": 0,
            "total_results": 0,
            "unique_sources": set(),
            "model_calls": 0,
        }

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now().isoformat()

    def _write_to_disk(self):
        """Write the current log structure to disk (thread-safe)."""
        with self.lock:
            try:
                # Ensure parent directory exists
                self.output_path.parent.mkdir(parents=True, exist_ok=True)

                # Convert Pydantic model to dict and write with pretty printing
                with open(self.output_path, 'w', encoding='utf-8') as f:
                    json.dump(self.log.model_dump(mode='json'), f, indent=2, ensure_ascii=False)
            except Exception as e:
                # Log error but don't crash the verification process
                print(f"Warning: Failed to write structured log to {self.output_path}: {e}")

    def log_claim(self, claim_id: str, claim_text: str, dataset: str, metadata: dict[str, Any] | None = None):
        """
        Log the initial claim information.

        Args:
            claim_id: Unique identifier for the claim.
            claim_text: The claim text to verify.
            dataset: Name of the dataset (e.g., "AVeriTeC", "MOCHEG").
            metadata: Additional dataset-specific metadata.
        """
        with self.lock:
            claim_data = {
                "id": claim_id,
                "text": claim_text,
                "dataset": dataset,
            }

            if metadata:
                # Add optional fields
                if "speaker" in metadata:
                    claim_data["speaker"] = metadata["speaker"]
                if "date" in metadata:
                    claim_data["date"] = metadata["date"]
                if "image_path" in metadata:
                    claim_data["image_path"] = metadata["image_path"]
                if "source" in metadata:
                    claim_data["source"] = metadata["source"]

                # Store any extra fields
                extra = {k: v for k, v in metadata.items()
                        if k not in ["speaker", "date", "image_path", "source"]}
                if extra:
                    claim_data["extra"] = extra

            self.log.claim = ClaimMetadata(**claim_data)

    def start_iteration(self, iteration_number: int):
        """
        Start a new iteration.

        Args:
            iteration_number: The iteration number (1-indexed).
        """
        with self.lock:
            self.iteration_start_time = time.time()

            iteration = IterationInfo(
                iteration_number=iteration_number,
                timestamp_start=self._get_timestamp(),
            )

            self.log.iterations.append(iteration)

    def log_planning(self, plan_text: str, actions_planned: list[str] | None = None, reasoning: str | None = None):
        """
        Log the planning phase.

        Args:
            plan_text: The full planning text/output.
            actions_planned: List of planned actions.
            reasoning: Optional reasoning behind the plan.
        """
        with self.lock:
            if not self.log.iterations:
                return

            planning = PlanningInfo(
                timestamp=self._get_timestamp(),
                plan_text=plan_text,
                actions_planned=actions_planned or [],
                reasoning=reasoning
            )

            self.log.iterations[-1].planning = planning

    def log_evidence_retrieval(
        self,
        action_type: str,
        tool: str,
        results: list[dict[str, Any]],
        query: str | None = None,
        platform: str | None = None,
        parameters: dict[str, Any] | None = None,
        execution_time: float | None = None,
        errors: list[str] | None = None,
    ):
        """
        Log an evidence retrieval action.

        Args:
            action_type: Type of action (e.g., "search", "geolocate").
            tool: Name of the tool used.
            results: List of results (each a dict with url, title, content, etc.).
            query: The query or input for the action.
            platform: Platform used (for searches: "google", "wikipedia", etc.).
            parameters: Additional parameters (date restrictions, etc.).
            execution_time: Time taken to execute the action (seconds).
            errors: List of error messages if any occurred.
        """
        with self.lock:
            if not self.log.iterations:
                return

            # Convert results to ResultInfo models
            result_infos: list[ResultInfo] = []
            unique_urls = set()

            for result in results:
                # Extract url (required field)
                url = result.get("url", "")
                if url:
                    unique_urls.add(url)

                # Create ResultInfo model
                result_info = ResultInfo(
                    url=url,
                    timestamp=self._get_timestamp(),
                    marked_useful=result.get("marked_useful", result.get("useful", False)),
                    title=result.get("title"),
                    snippet=result.get("snippet"),
                    content=result.get("content"),
                    summary=result.get("summary"),
                    platform=result.get("platform"),
                    error=result.get("error")
                )

                result_infos.append(result_info)

            # Create EvidenceRetrievalAction model
            action = EvidenceRetrievalAction(
                action_type=action_type,
                tool=tool,
                timestamp=self._get_timestamp(),
                results=result_infos,
                total_results=len(results),
                unique_results=len(unique_urls),
                errors=errors or [],
                query=query,
                platform=platform,
                parameters=parameters or {},
                execution_time_seconds=execution_time
            )

            self.log.iterations[-1].evidence_retrieval.append(action)

            # Update statistics
            if action_type == "search":
                self.stats["total_searches"] += 1
            self.stats["total_results"] += len(results)
            self.stats["unique_sources"].update(unique_urls)

    def log_elaboration(self, analysis_text: str, extracted_facts: list[str] | None = None, reasoning: str | None = None):
        """
        Log the elaboration phase.

        Args:
            analysis_text: The full elaboration/analysis text.
            extracted_facts: List of extracted facts.
            reasoning: Optional reasoning.
        """
        with self.lock:
            if not self.log.iterations:
                return

            elaboration = ElaborationInfo(
                timestamp=self._get_timestamp(),
                analysis_text=analysis_text,
                extracted_facts=extracted_facts or [],
                reasoning=reasoning
            )

            self.log.iterations[-1].elaboration = elaboration

    def log_judgment(self, reasoning: str, verdict: str, confidence: str | None = None):
        """
        Log the judgment phase.

        Args:
            reasoning: The reasoning behind the judgment.
            verdict: The verdict (e.g., "NEI", "SUPPORT", "REFUTE").
            confidence: Optional confidence level.
        """
        with self.lock:
            if not self.log.iterations:
                return

            judgment = JudgmentInfo(
                timestamp=self._get_timestamp(),
                reasoning=reasoning,
                verdict=verdict,
                confidence=confidence
            )

            self.log.iterations[-1].judgment = judgment

    def end_iteration(self):
        """
        End the current iteration and write to disk.
        """
        with self.lock:
            if not self.log.iterations:
                return

            # Set end timestamp
            self.log.iterations[-1].timestamp_end = self._get_timestamp()

        # Write to disk after each iteration
        self._write_to_disk()

    def log_final_verdict(self, label: str, justification: str, confidence: str | None = None):
        """
        Log the final verdict.

        Args:
            label: The final label.
            justification: Justification for the verdict.
            confidence: Optional confidence level.
        """
        with self.lock:
            verdict = FinalVerdict(
                label=label,
                justification=justification,
                timestamp=self._get_timestamp(),
                confidence=confidence
            )

            self.log.final_verdict = verdict

    def finalize(self):
        """
        Finalize the log with statistics and write final version to disk.
        """
        with self.lock:
            end_time = time.time()

            # Calculate statistics
            statistics = Statistics(
                total_iterations=len(self.log.iterations),
                total_searches=self.stats["total_searches"],
                total_results=self.stats["total_results"],
                unique_sources=len(self.stats["unique_sources"]),
                execution_time_seconds=end_time - self.start_time,
                model_calls=self.stats["model_calls"],
            )

            self.log.statistics = statistics
            self.log.timestamp_end = self._get_timestamp()

        # Final write to disk
        self._write_to_disk()

    def increment_model_calls(self, count: int = 1):
        """
        Increment the model call counter.

        Args:
            count: Number of model calls to add.
        """
        with self.lock:
            self.stats["model_calls"] += count
