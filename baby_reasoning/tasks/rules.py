from __future__ import annotations
import json
import random
from pathlib import Path

from baby_reasoning import DATA_DIR
from baby_reasoning.tasks.base import Condition, ModelResponse, Stimulus, Task

_SYLLABLES = [
    "ga",
    "ti",
    "li",
    "na",
    "ta",
    "da",
    "wo",
    "fe",
    "de",
    "ro",
    "ba",
    "fo",
    "bi",
    "ku",
    "me",
    "si",
    "pe",
    "zo",
    "re",
    "vi",
]

_RULES = ("ABA", "ABB", "AAB")


def _make_triplet(a: str, b: str, rule: str) -> tuple[str, str]:
    """Return (query_with_blank, expected) for a given rule."""
    if rule == "ABA":
        return f"{a} {b} ___", a
    elif rule == "ABB":
        return f"{a} {b} ___", b
    else:  # AAB
        return f"{a} ___ {b}", a


class RulesTask(Task):
    """Marcus (1999) abstract rule learning over nonsense syllable triplets."""

    _DATA_PATH = DATA_DIR / "rules" / "canonical.json"

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
        rule = random.choice(_RULES)
        pool = _SYLLABLES.copy()
        random.shuffle(pool)
        a, b = pool[0], pool[1]
        example_syllables = pool[2:]

        query, expected = _make_triplet(a, b, rule)

        # Build three few-shot examples using other syllables
        examples = []
        for i in range(3):
            ea, eb = example_syllables[i * 2], example_syllables[i * 2 + 1]
            ex_query, ex_answer = _make_triplet(ea, eb, rule)
            examples.append((ex_query, ex_answer))

        return Stimulus(
            query=query,
            expected=expected,
            few_shot_examples=examples,
            metadata={"rule": rule, "source": "generated"},
        )

    def score(self, response: ModelResponse, stimulus: Stimulus) -> bool:
        return response.text.strip().lower() == stimulus.expected.strip().lower()

    def build_prompt(self, stimulus: Stimulus, condition: Condition) -> str:
        lines = [
            "Complete the sequence by filling in the blank with a single word.",
            "",
        ]
        if condition == Condition.FEW_SHOT:
            lines.append("Examples:")
            for query, answer in stimulus.few_shot_examples:
                lines.append(f"  {query} → {answer}")
            lines.append("")
        lines.append(f"Now complete: {stimulus.query}")
        lines.append("Answer with only the missing word:")
        return "\n".join(lines)
