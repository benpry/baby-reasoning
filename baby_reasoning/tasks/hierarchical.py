from __future__ import annotations

import json
import random

from baby_reasoning import DATA_DIR
from baby_reasoning.tasks.base import ModelResponse, Stimulus, Task

_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_ANSWER_CHOICES = ["0", "1"]
_PATTERNS = {
    "same-same": (True, True),
    "same-different": (True, False),
    "different-different": (False, False),
}


def _random_pair(same: bool) -> str:
    """Return a random two-letter pair. Letters are unique within the pair."""
    a, b = random.sample(_LETTERS, 2)
    if same:
        return a * 2
    return a + b


def _make_query_and_examples(
    rel1: bool, rel2: bool, n_examples: int
) -> tuple[str, list[tuple[str, str]]]:
    """Build a query and independent few-shot examples.

    Each example draws its own letters so the pool is never exhausted.
    """
    query = _random_pair(rel1) + _random_pair(rel2)

    examples = []
    for _ in range(n_examples):
        e_rel1 = random.choice([True, False])
        e_rel2 = random.choice([True, False])
        e_query = _random_pair(e_rel1) + _random_pair(e_rel2)
        e_label = "1" if e_rel1 == e_rel2 else "0"
        examples.append((e_query, e_label))

    return query, examples


class HierarchicalTask(Task):
    """Hierarchical equality task (Marcus 2001 / Geiger 2022).

    Each stimulus is a pair of two-letter pairs concatenated with no spaces,
    e.g. "AABB" (same-same) or "ABCD" (different-different).
    The label is "1" when both pairs share the same Level-1 relation and
    "0" when they differ — discovered by the model purely through ICL.
    """

    _DATA_PATH = DATA_DIR / "hierarchical" / "canonical.json"

    def canonical_stimuli(self) -> list[Stimulus]:
        with open(self._DATA_PATH) as f:
            items = json.load(f)
        return [
            Stimulus(
                query=item["query"],
                expected=item["expected"],
                few_shot_examples=[tuple(e) for e in item.get("few_shot_examples", [])],
                metadata=item.get("metadata", {}),
                answer_choices=_ANSWER_CHOICES,
            )
            for item in items
        ]

    def generate_stimulus(self, n_examples: int = 3) -> Stimulus:
        rel1 = random.choice([True, False])
        rel2 = random.choice([True, False])
        label = "1" if rel1 == rel2 else "0"

        if rel1 == rel2:
            pattern = "same-same" if rel1 else "different-different"
        else:
            pattern = "same-different"

        query, examples = _make_query_and_examples(rel1, rel2, n_examples)

        return Stimulus(
            query=query,
            expected=label,
            few_shot_examples=examples,
            metadata={"pattern": pattern, "source": "generated"},
            answer_choices=_ANSWER_CHOICES,
        )

    def _generate_for_pattern(
        self, rel1: bool, rel2: bool, n_examples: int
    ) -> Stimulus:
        """Generate a stimulus with forced rel1/rel2 for the query."""
        label = "1" if rel1 == rel2 else "0"

        if rel1 == rel2:
            pattern = "same-same" if rel1 else "different-different"
        else:
            pattern = "same-different"

        query, examples = _make_query_and_examples(rel1, rel2, n_examples)

        return Stimulus(
            query=query,
            expected=label,
            few_shot_examples=examples,
            metadata={"pattern": pattern, "source": "systematic"},
            answer_choices=_ANSWER_CHOICES,
        )

    def systematic_stimuli(self, n_per_pattern: int, n_examples: int) -> list[Stimulus]:
        """Generate stimuli covering each pattern with ``n_per_pattern`` instances."""
        stimuli = []
        for _pattern_name, (rel1, rel2) in _PATTERNS.items():
            for _ in range(n_per_pattern):
                stimuli.append(self._generate_for_pattern(rel1, rel2, n_examples))
        return stimuli

    def format_completion(self, stimulus: Stimulus, choice: str) -> str:
        return " " + choice

    def score(self, response: ModelResponse, stimulus: Stimulus) -> bool:
        # check only the first character of the response
        return response.text.strip()[0] == stimulus.expected.strip()

    def build_prompt(self, stimulus: Stimulus, n_examples: int) -> str:
        if n_examples > 0 and stimulus.few_shot_examples:
            examples = stimulus.few_shot_examples[:n_examples]
            lines = [f"{q} {a}" for q, a in examples]
            lines.append(stimulus.query)
            return "\n".join(lines)
        return stimulus.query
