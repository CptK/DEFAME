from abc import ABC
from typing import Any
import time

import torch
from ezmm import MultimodalSequence

from defame.common import Action, Results, Evidence, MultimediaSnippet, Model, logger
from defame.common.structured_logger import StructuredLogger


class Tool(ABC):
    """Base class for all tools. Tools leverage integrations to retrieve evidence."""
    name: str
    actions: list[type(Action)]  # (classes of the) available actions this tool offers

    def __init__(self, llm: Model | None = None, device: str | torch.device | None = None):
        self.device = device
        self.llm = llm

        self.current_claim_id: str | None = None  # used by few tools to adjust claim-specific behavior

    def perform(
        self, action: Action, summarize: bool = True, structured_logger: StructuredLogger | None = None, **kwargs
    ) -> Evidence:
        assert type(action) in self.actions, f"Forbidden action: {action}"

        # Execute the action
        logger.log(f"[Tool:{self.name}] Starting _perform for {type(action).__name__}")
        start_time = time.time()
        try:
            result = self._perform(action, structured_logger=structured_logger)
            execution_time = time.time() - start_time
            logger.log(f"[Tool:{self.name}] _perform completed in {execution_time:.2f}s, got {len(result) if hasattr(result, '__len__') else '?'} results")
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"[Tool:{self.name}] _perform failed after {execution_time:.2f}s: {e}")
            raise

        # Summarize the result
        if summarize:
            logger.log(f"[Tool:{self.name}] Starting _summarize")
            start_time = time.time()
            try:
                summary = self._summarize(result, **kwargs)
                elapsed = time.time() - start_time
                logger.log(f"[Tool:{self.name}] _summarize completed in {elapsed:.2f}s")
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"[Tool:{self.name}] _summarize failed after {elapsed:.2f}s: {e}")
                raise
        else:
            summary = None

        evidence = Evidence(result, action, takeaways=summary)
        if structured_logger and not hasattr(self, '_log_search_results'):
            self._log_tool_action(structured_logger, action, result, evidence, execution_time)

        return evidence

    def _perform(self, action: Action, structured_logger: StructuredLogger | None = None) -> Results:
        """The actual function executing the action."""
        raise NotImplementedError

    def _summarize(self, result: Results, **kwargs) -> MultimodalSequence | None:
        """Turns the result into an LLM-friendly summary. May use additional
        context for summarization. Returns None iff the result does not contain any
        (potentially) helpful information."""
        raise NotImplementedError

    def reset(self) -> None:
        """Resets the tool to its initial state (if applicable) and sets all stats to zero."""
        pass

    def get_stats(self) -> dict[str, Any]:
        """Returns the tool's usage statistics as a dictionary."""
        return {}

    def set_claim_id(self, claim_id: str):
        self.current_claim_id = claim_id

    def _log_tool_action(self, structured_logger, action: Action, result: Results,
                        evidence: Evidence, execution_time: float):
        """Default logging implementation for tools."""
        # Extract basic result information
        result_dict = {
            "action": str(action),
            "result_type": type(result).__name__,
        }

        # Add summary if available
        if evidence.takeaways:
            result_dict["summary"] = str(evidence.takeaways)

        # Try to extract useful flag
        result_dict["useful"] = evidence.is_useful() if hasattr(evidence, 'is_useful') else False

        # Log the action
        structured_logger.log_evidence_retrieval(
            action_type=action.name if hasattr(action, 'name') else type(action).__name__,
            tool=self.name,
            results=[result_dict],
            execution_time=execution_time
        )


def get_available_actions(tools: list[Tool], available_actions: list[Action] | None) -> set[type[Action]]:
    actions = set()
    for tool in tools:
        actions.update(tool.actions)
    if available_actions is not None:
        actions = actions.intersection(set(available_actions))
    return actions
