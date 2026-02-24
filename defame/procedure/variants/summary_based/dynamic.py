from typing import Any

from defame.common import Report, Label, logger
from defame.common.structured_logger import StructuredLogger
from defame.procedure.procedure import Procedure


class DynamicSummary(Procedure):
    def __init__(self, max_iterations: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.max_iterations = max_iterations

    def apply_to(
        self, doc: Report, structured_logger: StructuredLogger | None = None
    ) -> tuple[Label, dict[str, Any]]:
        n_iterations = 0
        label = Label.NEI
        while label == Label.NEI and n_iterations < self.max_iterations:
            if n_iterations > 0:
                logger.log("Not enough information yet. Continuing fact-check...")
            n_iterations += 1

            # Start logging this iteration
            if structured_logger:
                structured_logger.start_iteration(n_iterations)

            # Planning phase
            actions, reasoning = self.planner.plan_next_actions(doc)
            if len(reasoning) > 32:  # Only keep substantial reasoning
                doc.add_reasoning(reasoning)

            # Log planning
            if structured_logger:
                action_strs = [str(a) for a in actions] if actions else []
                structured_logger.log_planning(
                    plan_text=reasoning,
                    actions_planned=action_strs
                )

            # Evidence retrieval phase
            if actions:
                doc.add_actions(actions)
                # Pass structured logger to actor for logging evidence retrieval
                evidences = self.actor.perform(actions, doc, structured_logger=structured_logger)
                doc.add_evidence(evidences)  # even if no evidence, add empty evidence block for the record

                # Elaboration phase
                elaboration_reasoning = self._develop(doc)

                # Log elaboration
                if structured_logger and elaboration_reasoning:
                    structured_logger.log_elaboration(analysis_text=elaboration_reasoning)

            # Judgment phase
            label = self.judge.judge(doc, is_final=n_iterations == self.max_iterations or not actions)

            # Log judgment
            if structured_logger:
                judgment_reasoning = self.judge.get_latest_reasoning()
                structured_logger.log_judgment(
                    reasoning=judgment_reasoning,
                    verdict=label.name
                )

            # End iteration
            if structured_logger:
                structured_logger.end_iteration()

        return label, {}
