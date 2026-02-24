"""
Blueprint system for DEFAME fact-checking.

Blueprints define verification strategies tailored to different types of claims.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml


@dataclass
class BlueprintAction:
    """Single action within a blueprint iteration."""
    action: str
    params: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: dict) -> "BlueprintAction":
        """Create BlueprintAction from dictionary."""
        if isinstance(data, str):
            # Handle simple string action (e.g., "search")
            return cls(action=data, params=None)
        return cls(
            action=data["action"],
            params=data.get("params")
        )


@dataclass
class BlueprintIteration:
    """Single iteration in a blueprint."""
    iteration: int
    actions: List[BlueprintAction]
    synthesis: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "BlueprintIteration":
        """Create BlueprintIteration from dictionary."""
        actions = [BlueprintAction.from_dict(a) for a in data["actions"]]
        return cls(
            iteration=data["iteration"],
            actions=actions,
            synthesis=data.get("synthesis", False)
        )


@dataclass
class StoppingCriteria:
    """Stopping criteria for blueprint execution."""
    max_iterations: int
    early_stop_conditions: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "StoppingCriteria":
        """Create StoppingCriteria from dictionary."""
        return cls(
            max_iterations=data["max_iterations"],
            early_stop_conditions=data.get("early_stop_conditions", [])
        )


@dataclass
class Blueprint:
    """
    A verification blueprint defining a strategy for fact-checking.

    Attributes:
        name: Unique identifier for the blueprint
        description: Human-readable description
        claim_characteristics: Types of claims this blueprint handles
        iterations: Sequence of verification steps
        stopping_criteria: When to stop verification
        rationale: Explanation of the strategy (optional)
    """
    name: str
    description: str
    claim_characteristics: List[str]
    iterations: List[BlueprintIteration]
    stopping_criteria: StoppingCriteria
    rationale: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "Blueprint":
        """
        Load a blueprint from a YAML file.

        Args:
            path: Path to the YAML blueprint file

        Returns:
            Blueprint instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML is malformed or missing required fields
        """
        if not path.exists():
            raise FileNotFoundError(f"Blueprint file not found: {path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Validate required fields
        required = ["name", "description", "claim_characteristics", "iterations", "stopping_criteria"]
        missing = [f for f in required if f not in data]
        if missing:
            raise ValueError(f"Blueprint missing required fields: {missing}")

        # Parse iterations
        iterations = [BlueprintIteration.from_dict(it) for it in data["iterations"]]

        # Parse stopping criteria
        stopping_criteria = StoppingCriteria.from_dict(data["stopping_criteria"])

        return cls(
            name=data["name"],
            description=data["description"],
            claim_characteristics=data["claim_characteristics"],
            iterations=iterations,
            stopping_criteria=stopping_criteria,
            rationale=data.get("rationale")
        )

    def get_iteration(self, iteration_num: int) -> Optional[BlueprintIteration]:
        """
        Get the actions for a specific iteration.

        Args:
            iteration_num: Iteration number (1-indexed)

        Returns:
            BlueprintIteration if exists, None otherwise
        """
        for it in self.iterations:
            if it.iteration == iteration_num:
                return it
        return None

    def get_max_iterations(self) -> int:
        """Get maximum number of iterations for this blueprint."""
        return self.stopping_criteria.max_iterations

    def should_synthesize(self, iteration_num: int) -> bool:
        """
        Check if synthesis should occur after this iteration.

        Args:
            iteration_num: Iteration number (1-indexed)

        Returns:
            True if synthesis should occur, False otherwise
        """
        it = self.get_iteration(iteration_num)
        return it.synthesis if it else False

    def to_dict(self) -> dict:
        """Convert blueprint to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "claim_characteristics": self.claim_characteristics,
            "iterations": [
                {
                    "iteration": it.iteration,
                    "actions": [
                        {"action": a.action, "params": a.params} if a.params
                        else a.action
                        for a in it.actions
                    ],
                    "synthesis": it.synthesis
                }
                for it in self.iterations
            ],
            "stopping_criteria": {
                "max_iterations": self.stopping_criteria.max_iterations,
                "early_stop_conditions": self.stopping_criteria.early_stop_conditions
            },
            "rationale": self.rationale
        }

    def __repr__(self) -> str:
        return f"Blueprint(name='{self.name}', iterations={len(self.iterations)})"


class BlueprintRegistry:
    """Registry for managing available blueprints."""

    def __init__(self, blueprints_dir: Optional[Path] = None):
        """
        Initialize blueprint registry.

        Args:
            blueprints_dir: Directory containing blueprint YAML files.
                           Defaults to config/blueprints/
        """
        if blueprints_dir is None:
            # Default to config/blueprints relative to project root
            project_root = Path(__file__).parent.parent.parent
            blueprints_dir = project_root / "config" / "blueprints"

        self.blueprints_dir = Path(blueprints_dir)
        self._blueprints: Dict[str, Blueprint] = {}

        # Load all blueprints
        if self.blueprints_dir.exists():
            self.load_all()

    def load_all(self) -> None:
        """Load all blueprint YAML files from the blueprints directory."""
        for yaml_file in self.blueprints_dir.glob("*.yaml"):
            # Skip schema file
            if yaml_file.name == "schema.yaml":
                continue

            try:
                blueprint = Blueprint.from_yaml(yaml_file)
                self._blueprints[blueprint.name] = blueprint
            except Exception as e:
                print(f"Warning: Failed to load blueprint {yaml_file.name}: {e}")

    def get(self, name: str) -> Optional[Blueprint]:
        """
        Get a blueprint by name.

        Args:
            name: Blueprint name

        Returns:
            Blueprint if found, None otherwise
        """
        return self._blueprints.get(name)

    def list_blueprints(self) -> List[str]:
        """Get list of all available blueprint names."""
        return list(self._blueprints.keys())

    def get_all(self) -> Dict[str, Blueprint]:
        """Get all blueprints as a dictionary."""
        return self._blueprints.copy()

    def add(self, blueprint: Blueprint) -> None:
        """
        Add a blueprint to the registry.

        Args:
            blueprint: Blueprint to add
        """
        self._blueprints[blueprint.name] = blueprint

    def __len__(self) -> int:
        return len(self._blueprints)

    def __repr__(self) -> str:
        return f"BlueprintRegistry({len(self)} blueprints: {self.list_blueprints()})"
