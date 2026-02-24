"""
Blueprint selection for DEFAME fact-checking.

Selects appropriate verification strategy (blueprint) based on claim characteristics.
"""

import json
import re
from typing import List, Dict, Optional
from defame.common import Claim, Model, Prompt, logger
from defame.common.blueprint import Blueprint, BlueprintRegistry


class BlueprintSelector:
    """
    Selects appropriate blueprint for a claim.

    Supports multiple selection strategies:
    - rule_based: Simple rules based on claim features
    - llm_based: LLM analyzes claim and selects blueprint
    - hybrid: Combination of rules and LLM
    """

    def __init__(self,
                 registry: BlueprintRegistry,
                 llm: Optional[Model] = None,
                 strategy: str = "llm_based"):
        """
        Initialize blueprint selector.

        Args:
            registry: Blueprint registry containing available blueprints
            llm: Language model for LLM-based selection
            strategy: Selection strategy - "rule_based", "llm_based", or "hybrid"
        """
        self.registry = registry
        self.llm = llm
        self.strategy = strategy

        if strategy in ["llm_based", "hybrid"] and llm is None:
            raise ValueError(f"LLM required for {strategy} strategy")

        logger.info(f"Initialized BlueprintSelector with {strategy} strategy")

    def select_blueprint(self, claim: Claim) -> Blueprint:
        """
        Select the most appropriate blueprint for a claim.

        Args:
            claim: Claim to analyze

        Returns:
            Selected Blueprint

        Raises:
            ValueError: If no suitable blueprint found
        """
        if self.strategy == "rule_based":
            blueprint_name = self._rule_based_selection(claim)
        elif self.strategy == "llm_based":
            blueprint_name = self._llm_based_selection(claim)
        elif self.strategy == "hybrid":
            blueprint_name = self._hybrid_selection(claim)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        blueprint = self.registry.get(blueprint_name)
        if blueprint is None:
            logger.warning(f"Blueprint '{blueprint_name}' not found, using fallback")
            blueprint = self._get_fallback_blueprint()

        logger.info(f"Selected blueprint '{blueprint.name}' for claim")
        return blueprint

    def _rule_based_selection(self, claim: Claim) -> str:
        """
        Select blueprint using simple rule-based heuristics.

        Args:
            claim: Claim to analyze

        Returns:
            Blueprint name
        """
        features = self._extract_claim_features(claim)

        # Priority-based rule matching
        # 1. Visual misinformation (if image and certain keywords)
        if features["has_image"]:
            if any(kw in str(claim).lower() for kw in [
                "fake", "deepfake", "photoshop", "manipulated", "edited",
                "authentic", "real", "doctored", "altered"
            ]):
                return "visual_misinformation"

            # 2. Complex multimodal (if image and complex claim)
            if features["is_complex"]:
                return "complex_multimodal"

        # 3. Quote verification
        if features["has_quote"]:
            return "quote_verification"

        # 4. Numerical claims
        if features["has_numbers"]:
            return "numerical_claim"

        # 5. Temporal verification
        if features["has_date"] or features["is_temporal"]:
            return "temporal_verification"

        # 6. Source comparison (controversial keywords)
        if features["is_controversial"]:
            return "source_comparison"

        # 7. Deep investigation (complex claims)
        if features["is_complex"]:
            return "deep_investigation"

        # 8. Simple factual (default)
        return "simple_factual"

    def _llm_based_selection(self, claim: Claim) -> str:
        """
        Select blueprint using LLM analysis.

        Args:
            claim: Claim to analyze

        Returns:
            Blueprint name
        """
        # Get available blueprints
        available_blueprints = self.registry.get_all()

        # Create prompt
        prompt = self._create_selection_prompt(claim, available_blueprints)

        # Query LLM
        try:
            response = self.llm.generate(prompt)
            if response is None:
                logger.warning("LLM returned None, falling back to rule-based")
                return self._rule_based_selection(claim)

            # Extract blueprint name from JSON response
            # Strip markdown code fences if present
            response_text = response.strip()
            if response_text.startswith("```"):
                response_text = re.sub(r"^```(?:json)?\s*", "", response_text)
                response_text = re.sub(r"\s*```$", "", response_text)
            parsed = json.loads(response_text)
            blueprint_name = parsed.get("blueprint_name")

            if blueprint_name is None:
                logger.warning("No blueprint_name in LLM response, falling back")
                return self._rule_based_selection(claim)

            return blueprint_name

        except Exception as e:
            logger.warning(f"LLM selection failed: {e}, falling back to rule-based")
            return self._rule_based_selection(claim)

    def _hybrid_selection(self, claim: Claim) -> str:
        """
        Hybrid selection: use rules first, then LLM for uncertain cases.

        Args:
            claim: Claim to analyze

        Returns:
            Blueprint name
        """
        features = self._extract_claim_features(claim)

        # Use rules for clear-cut cases
        # Visual content with manipulation keywords
        if features["has_image"] and any(kw in str(claim).lower() for kw in [
            "deepfake", "photoshop", "manipulated", "doctored", "altered"
        ]):
            return "visual_misinformation"

        # Clear quotes
        if features["has_quote"] and features["mentions_person"]:
            return "quote_verification"

        # For less certain cases, use LLM
        return self._llm_based_selection(claim)

    def _extract_claim_features(self, claim: Claim) -> Dict[str, bool]:
        """
        Extract features from claim for rule-based selection.

        Args:
            claim: Claim to analyze

        Returns:
            Dictionary of boolean features
        """
        text = str(claim).lower()

        # Check for images
        has_image = claim.has_images()

        # Check for quotes
        has_quote = bool(re.search(r'["\'].*["\']', text)) or \
                   any(word in text for word in ["said", "stated", "claimed", "announced"])

        # Check for person mentions (simple heuristic)
        mentions_person = bool(re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+', str(claim)))

        # Check for numbers
        has_numbers = bool(re.search(r'\d+', text))

        # Check for dates
        date_keywords = ["january", "february", "march", "april", "may", "june",
                        "july", "august", "september", "october", "november", "december",
                        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        has_date = any(kw in text for kw in date_keywords) or \
                  bool(re.search(r'\d{4}', text)) or claim.date is not None

        # Check for temporal keywords
        temporal_keywords = ["before", "after", "during", "when", "since", "until",
                           "first", "last", "ago", "recent", "history"]
        is_temporal = any(kw in text for kw in temporal_keywords)

        # Check for controversial keywords
        controversial_keywords = ["allegedly", "reportedly", "claims", "dispute",
                                "controversy", "debate", "conflicting", "accused"]
        is_controversial = any(kw in text for kw in controversial_keywords)

        # Check complexity (simple heuristic based on length and structure)
        is_complex = len(text) > 200 or text.count(",") > 3 or text.count(";") > 0

        return {
            "has_image": has_image,
            "has_quote": has_quote,
            "mentions_person": mentions_person,
            "has_numbers": has_numbers,
            "has_date": has_date,
            "is_temporal": is_temporal,
            "is_controversial": is_controversial,
            "is_complex": is_complex
        }

    def _create_selection_prompt(self,
                                 claim: Claim,
                                 available_blueprints: Dict[str, Blueprint]) -> Prompt:
        """
        Create prompt for LLM-based blueprint selection.

        Args:
            claim: Claim to analyze
            available_blueprints: Available blueprints

        Returns:
            Prompt for LLM
        """
        # Build blueprint descriptions
        blueprint_descriptions = []
        for name, bp in available_blueprints.items():
            blueprint_descriptions.append(
                f"- **{name}**: {bp.description}\n"
                f"  Characteristics: {', '.join(bp.claim_characteristics)}"
            )

        has_images = claim.has_images()
        image_note = ("IMPORTANT: This claim has NO attached images or videos. "
                      "Do NOT select visual/multimodal blueprints." if not has_images
                      else "This claim includes attached images.")

        prompt_text = f"""You are a fact-checking strategy selector. Given a claim, select the most appropriate verification blueprint.

CLAIM:
{claim}

{image_note}

AVAILABLE BLUEPRINTS:
{chr(10).join(blueprint_descriptions)}

Analyze the claim and select the single most appropriate blueprint. Consider:
1. Does the claim actually include attached images or videos? (A claim merely *mentioning* photos does NOT count as having images.)
2. Is it a quote or statement attributed to someone?
3. Does it involve numbers, statistics, or quantitative data?
4. Is temporal context important (dates, events, timeline)?
5. Is it controversial or likely to have conflicting sources?
6. How complex is the claim (single fact vs. multi-faceted)?

Respond with JSON:
{{
    "analysis": "Brief analysis of claim characteristics",
    "blueprint_name": "selected_blueprint_name",
    "reasoning": "Why this blueprint is most appropriate"
}}
"""

        return Prompt(text=prompt_text)

    def _get_fallback_blueprint(self) -> Blueprint:
        """
        Get fallback blueprint when selection fails.

        Returns:
            Fallback blueprint (simple_factual or first available)
        """
        # Try to get simple_factual as default
        fallback = self.registry.get("simple_factual")
        if fallback:
            return fallback

        # Otherwise return first available
        all_blueprints = self.registry.get_all()
        if all_blueprints:
            return list(all_blueprints.values())[0]

        raise ValueError("No blueprints available in registry")

    def batch_select(self, claims: List[Claim]) -> List[Blueprint]:
        """
        Select blueprints for multiple claims (useful for experiments).

        Args:
            claims: List of claims

        Returns:
            List of selected blueprints (same order as claims)
        """
        return [self.select_blueprint(claim) for claim in claims]

    def get_selection_statistics(self, claims: List[Claim]) -> Dict[str, int]:
        """
        Get statistics on blueprint selection for a set of claims.

        Args:
            claims: List of claims

        Returns:
            Dictionary mapping blueprint names to count
        """
        blueprints = self.batch_select(claims)
        stats = {}
        for bp in blueprints:
            stats[bp.name] = stats.get(bp.name, 0) + 1
        return stats
