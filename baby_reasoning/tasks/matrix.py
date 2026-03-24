from __future__ import annotations
import json
import random

from baby_reasoning import DATA_DIR
from baby_reasoning.tasks.base import Condition, ModelResponse, Stimulus, Task

_SHAPES = ["circle", "triangle", "square", "pentagon", "hexagon"]
_SIZES = ["large", "medium", "small"]
_COLORS = ["red", "blue", "green", "yellow", "purple"]


def _rotation_matrix(items: list[str]) -> tuple[str, str]:
    """Generate a 3x3 rotation matrix query and expected answer."""
    a, b, c = items[0], items[1], items[2]
    rows = [
        f"Row 1: {a} | {b} | {c}",
        f"Row 2: {b} | {c} | {a}",
        f"Row 3: {c} | {a} | ___",
    ]
    return "\n".join(rows), b


class MatrixTask(Task):
    """Text-encoded Raven's Progressive Matrices (Webb et al. 2023)."""

    _DATA_PATH = DATA_DIR / "matrix" / "canonical.json"

    def canonical_stimuli(self) -> list[Stimulus]:
        with open(self._DATA_PATH) as f:
            items = json.load(f)
        return [
            Stimulus(
                query=item["query"],
                expected=item["expected"],
                few_shot_examples=[tuple(e) for e in item.get("few_shot_examples", [])],
                metadata=item.get("metadata", {}),
            )
            for item in items
        ]

    def generate_stimulus(self) -> Stimulus:
        shapes = random.sample(_SHAPES, 3)
        query, expected = _rotation_matrix(shapes)
        return Stimulus(
            query=query,
            expected=expected,
            few_shot_examples=[],
            metadata={"rule": "rotation", "attributes": ["shape"], "source": "generated"},
        )

    def score(self, response: ModelResponse, stimulus: Stimulus) -> bool:
        return response.text.strip().lower() == stimulus.expected.strip().lower()

    def build_prompt(self, stimulus: Stimulus, condition: Condition) -> str:
        lines = [
            "Complete the pattern matrix by filling in the missing cell (marked ___).",
            "Answer with only the missing item.",
            "",
        ]
        if condition == Condition.FEW_SHOT and stimulus.few_shot_examples:
            lines.append("Examples:")
            for query, answer in stimulus.few_shot_examples:
                indented = query.replace("\n", "\n    ")
                lines.append(f"  Matrix:\n    {indented}")
                lines.append(f"  Answer: {answer}")
                lines.append("")
        lines.append("Matrix:")
        lines.append(stimulus.query)
        lines.append("Answer:")
        return "\n".join(lines)
