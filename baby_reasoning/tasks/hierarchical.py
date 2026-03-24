from __future__ import annotations

import json
import random

from baby_reasoning import DATA_DIR
from baby_reasoning.tasks.base import Condition, ModelResponse, Stimulus, Task

_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class HierarchicalTask(Task):
    """Hierarchical equality task: judge whether two pairs share the same Level-1 relation."""

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
        meta_rel = "same" if rel1 == rel2 else "different"

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
            e_answer = "same" if e_rel1 == e_rel2 else "different"
            examples.append((f"{e_pair1} {e_pair2}", e_answer))

        return Stimulus(
            query=f"{pair1} {pair2}",
            expected=meta_rel,
            few_shot_examples=examples,
            metadata={"pattern": pattern, "source": "generated"},
        )

    def score(self, response: ModelResponse, stimulus: Stimulus) -> bool:
        return response.text.strip().lower() == stimulus.expected.strip().lower()

    def build_prompt(self, stimulus: Stimulus, condition: Condition) -> str:
        lines = [
            "Two pairs of letters are shown. Each pair is either 'same' (both letters identical)",
            "or 'different' (letters differ). Judge whether the two pairs share the SAME",
            "first-level relationship, or have DIFFERENT first-level relationships.",
            "Answer with only 'same' or 'different'.",
            "",
        ]
        if condition == Condition.FEW_SHOT:
            lines.append("Examples:")
            for query, answer in stimulus.few_shot_examples:
                lines.append(f"  {query} → {answer}")
            lines.append("")
        lines.append(f"Pairs: {stimulus.query}")
        lines.append("Answer:")
        return "\n".join(lines)
