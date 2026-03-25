from __future__ import annotations

import json
import random

from baby_reasoning import DATA_DIR
from baby_reasoning.tasks.base import Condition, ModelResponse, Stimulus, Task

_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_ANSWER_CHOICES = ["0", "1"]


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

    def generate_stimulus(self) -> Stimulus:
        pool = _LETTERS.copy()
        random.shuffle(pool)
        letter_idx = 0

        def next_pair(same: bool) -> str:
            nonlocal letter_idx
            if same:
                pair = pool[letter_idx] * 2
                letter_idx += 1
            else:
                pair = pool[letter_idx] + pool[letter_idx + 1]
                letter_idx += 2
            return pair

        rel1 = random.choice([True, False])
        rel2 = random.choice([True, False])
        pair1 = next_pair(rel1)
        pair2 = next_pair(rel2)
        label = "1" if rel1 == rel2 else "0"

        if rel1 == rel2:
            pattern = "same-same" if rel1 else "different-different"
        else:
            pattern = "same-different"

        examples = []
        for _ in range(3):
            e_rel1 = random.choice([True, False])
            e_rel2 = random.choice([True, False])
            e_pair1 = next_pair(e_rel1)
            e_pair2 = next_pair(e_rel2)
            e_label = "1" if e_rel1 == e_rel2 else "0"
            examples.append((e_pair1 + e_pair2, e_label))

        return Stimulus(
            query=pair1 + pair2,
            expected=label,
            few_shot_examples=examples,
            metadata={"pattern": pattern, "source": "generated"},
            answer_choices=_ANSWER_CHOICES,
        )

    def score(self, response: ModelResponse, stimulus: Stimulus) -> bool:
        return response.text.strip() == stimulus.expected.strip()

    def build_prompt(self, stimulus: Stimulus, condition: Condition) -> str:
        if condition == Condition.FEW_SHOT:
            lines = [f"{q} {a}" for q, a in stimulus.few_shot_examples]
            lines.append(stimulus.query)
            return "\n".join(lines)
        return stimulus.query
