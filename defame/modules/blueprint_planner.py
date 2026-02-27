"""
Blueprint-based planner for DEFAME fact-checking.

Uses predefined verification strategies (blueprints) instead of LLM-based planning.
"""

from typing import Collection, Optional
from defame.common.action import Action
from defame.common import logger, Report, Model
from defame.common.blueprint import Blueprint
from defame.modules.planner import Planner


class BlueprintPlanner(Planner):
    """
    Planner that follows a predefined blueprint strategy.

    Instead of using LLM to plan actions, follows the action sequence
    defined in a Blueprint. This allows for consistent, reproducible
    verification strategies tailored to different claim types.
    """

    def __init__(self,
                 valid_actions: Collection[type[Action]],
                 llm: Model,
                 extra_rules: str,
                 blueprint: Blueprint,
                 mode: str = "pure"):
        """
        Initialize BlueprintPlanner.

        Args:
            valid_actions: Collection of valid Action types
            llm: Language model for generating reasoning (if mode != "pure")
            extra_rules: Extra rules to pass to LLM (if mode != "pure")
            blueprint: Blueprint defining the verification strategy
            mode: Planning mode - "pure", "guided", or "llm"
                  - "pure": Follow blueprint exactly, no LLM involved
                  - "guided": Use blueprint actions but LLM generates reasoning
                  - "llm": Fallback to normal LLM planning (useful for comparison)
        """
        super().__init__(valid_actions, llm, extra_rules)
        self.blueprint = blueprint
        self.mode = mode

        # Validate mode
        if mode not in ["pure", "guided", "llm"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'pure', 'guided', or 'llm'")

        logger.info(f"Initialized BlueprintPlanner with blueprint '{blueprint.name}' in '{mode}' mode")

    def plan_next_actions(self, doc: Report, all_actions=False) -> tuple[list[Action], str]:
        """
        Plan next actions based on blueprint.

        Args:
            doc: Current fact-checking report
            all_actions: If True, return all remaining blueprint actions (unused)

        Returns:
            Tuple of (actions, reasoning)
        """
        # If mode is "llm", use parent's LLM-based planning
        if self.mode == "llm":
            return super().plan_next_actions(doc, all_actions)

        current_iteration = doc.get_iteration()

        # Check if we've exceeded max iterations
        if current_iteration > self.blueprint.get_max_iterations():
            logger.info(f"Reached max iterations ({self.blueprint.get_max_iterations()})")
            return [], "Maximum iterations reached according to blueprint"

        # Get blueprint iteration
        blueprint_iteration = self.blueprint.get_iteration(current_iteration)

        if blueprint_iteration is None:
            logger.info(f"No blueprint defined for iteration {current_iteration}")
            return [], f"No actions defined for iteration {current_iteration}"

        if self.mode == "guided":
            # Use LLM to generate concrete actions, guided by the blueprint structure
            return self._guided_plan(doc, blueprint_iteration, current_iteration)

        # Pure mode: convert blueprint actions directly to Action objects
        actions = self._convert_blueprint_actions(blueprint_iteration.actions, doc)

        # Filter out already-performed actions
        performed_actions = doc.get_all_actions()
        actions = [action for action in actions if action not in performed_actions]

        reasoning = self._generate_blueprint_reasoning(
            current_iteration,
            blueprint_iteration,
            actions
        )

        logger.info(f"Blueprint planner (iteration {current_iteration}): {len(actions)} actions planned")

        return actions, reasoning

    def _convert_blueprint_actions(self, blueprint_actions, doc: Report) -> list[Action]:
        """
        Convert blueprint action specifications to Action objects.

        Args:
            blueprint_actions: List of BlueprintAction from blueprint
            doc: Current fact-checking report

        Returns:
            List of Action objects
        """
        actions = []
        action_lookup = {action.name: action for action in self.valid_actions}

        for bp_action in blueprint_actions:
            action_name = bp_action.action.lower()

            # Find matching action class
            action_class = None
            for valid_action in self.valid_actions:
                if valid_action.name.lower() == action_name:
                    action_class = valid_action
                    break

            if action_class is None:
                logger.warning(f"Blueprint action '{bp_action.action}' not in valid_actions, skipping")
                continue

            # Create action instance with params
            try:
                if bp_action.params:
                    action = action_class(**bp_action.params)
                else:
                    action = action_class()
                actions.append(action)
            except Exception as e:
                logger.warning(f"Failed to create action {action_name}: {e}")

        return actions

    def _generate_blueprint_reasoning(self,
                                     iteration: int,
                                     blueprint_iteration,
                                     actions: list[Action]) -> str:
        """
        Generate reasoning text explaining why these actions were chosen.

        Args:
            iteration: Current iteration number
            blueprint_iteration: BlueprintIteration object
            actions: List of actions to perform

        Returns:
            Reasoning text
        """
        action_names = [type(a).name for a in actions]

        reasoning = f"Following blueprint '{self.blueprint.name}' (iteration {iteration}).\n"
        reasoning += f"This blueprint is designed for: {self.blueprint.description}\n"
        reasoning += f"Planned actions: {', '.join(action_names)}\n"

        if blueprint_iteration.synthesis:
            reasoning += "After these actions, a synthesis stage will integrate the evidence."

        return reasoning

    def _guided_plan(self, doc: Report, blueprint_iteration, current_iteration: int) -> tuple[list[Action], str]:
        """Use the LLM to plan concrete actions, guided by the blueprint's structure."""
        from defame.evidence_retrieval.tools import IMAGE_ACTIONS
        image_action_names = {a.name for a in IMAGE_ACTIONS}
        has_images = doc.claim.has_images()

        # Build guidance from the blueprint iteration, skipping image actions if no images
        # and skipping actions not in valid_actions
        valid_action_names = {a.name for a in self.valid_actions}
        action_counts = {}
        for bp_action in blueprint_iteration.actions:
            name = bp_action.action.lower()
            if not has_images and name in image_action_names:
                continue
            if name not in valid_action_names:
                continue
            action_counts[name] = action_counts.get(name, 0) + 1

        if not action_counts:
            # All blueprint actions were image-based but claim has no images; fall back to search
            action_counts["search"] = 1

        guidance_parts = []
        for action_name, count in action_counts.items():
            guidance_parts.append(f"{count}x {action_name}")
        actions_summary = ", ".join(guidance_parts)

        blueprint_rules = (
            f"You are following the '{self.blueprint.name}' blueprint (iteration {current_iteration}). "
            f"This blueprint is designed for: {self.blueprint.description}\n"
            f"For this iteration, plan exactly these actions: {actions_summary}. "
            f"Generate appropriate queries/parameters for each action based on the claim and evidence so far."
        )
        if not has_images:
            blueprint_rules += "\nIMPORTANT: This claim has NO images. Do NOT use any image-based actions."

        # Temporarily augment extra_rules and use the parent's LLM-based planning
        original_extra_rules = self.extra_rules
        self.extra_rules = blueprint_rules if not self.extra_rules else f"{self.extra_rules}\n{blueprint_rules}"
        try:
            actions, reasoning = super().plan_next_actions(doc)
        finally:
            self.extra_rules = original_extra_rules

        logger.info(f"Blueprint planner guided (iteration {current_iteration}): {len(actions)} actions planned")
        return actions, reasoning

    def _generate_llm_reasoning(self, doc: Report, actions: list[Action]) -> str:
        """
        Use LLM to generate reasoning for the planned actions.

        Args:
            doc: Current fact-checking report
            actions: List of actions to perform

        Returns:
            LLM-generated reasoning text
        """
        # TODO: Create a prompt that asks LLM to explain why these specific
        # actions are appropriate given the current state of the report
        # For now, use blueprint reasoning
        return self._generate_blueprint_reasoning(
            doc.get_iteration(),
            self.blueprint.get_iteration(doc.get_iteration()),
            actions
        )

    def should_stop(self, doc: Report, confidence: Optional[float] = None) -> tuple[bool, str]:
        """
        Determine if verification should stop based on blueprint criteria.

        Args:
            doc: Current fact-checking report
            confidence: Current confidence level (if available)

        Returns:
            Tuple of (should_stop, reason)
        """
        current_iteration = doc.get_iteration()

        # Check max iterations
        if current_iteration >= self.blueprint.get_max_iterations():
            return True, f"Reached maximum iterations ({self.blueprint.get_max_iterations()})"

        # Check early stopping conditions
        for condition in self.blueprint.stopping_criteria.early_stop_conditions:
            condition_type = condition.get("type")

            if condition_type == "high_confidence" and confidence is not None:
                threshold = condition.get("threshold", 0.9)
                if confidence >= threshold:
                    return True, f"High confidence reached ({confidence:.2f} >= {threshold})"

            elif condition_type == "no_new_evidence":
                # Check if last iteration produced no new evidence
                # This would need access to evidence tracking
                # For now, skip this condition
                pass

            elif condition_type == "contradiction_found":
                # Check if contradictions have been found in evidence
                # This would need access to evidence analysis
                # For now, skip this condition
                pass

        return False, ""
