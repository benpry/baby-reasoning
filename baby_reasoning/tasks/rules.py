from __future__ import annotations

import json
import random

from baby_reasoning import DATA_DIR
from baby_reasoning.tasks.base import ModelResponse, Stimulus, Task

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

_RULES = ("ABA", "ABB")


def _make_triplet(a: str, b: str, rule: str) -> tuple[str, str, str]:
    """Return (syllable_a, syllable_b, expected_third) for a given rule."""
    if rule == "ABA":
        return a, b, a
    else:  # ABB
        return a, b, b


class RulesTask(Task):
    """Marcus (1999) abstract rule learning over nonsense syllable triplets.

    Stimuli are formatted as sequence completions: two syllables shown,
    model predicts the third. No blank placeholders. In few-shot condition,
    complete triplets are given as context lines (base-model ICL style).
    """

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
                answer_choices=item.get("answer_choices"),
            )
            for item in items
        ]

    def generate_stimulus(self, n_examples: int = 3) -> Stimulus:
        rule = random.choice(_RULES)
        pool = _SYLLABLES.copy()
        random.shuffle(pool)
        a, b, expected = _make_triplet(pool[0], pool[1], rule)

        examples = []
        for _ in range(n_examples):
            ex_pool = _SYLLABLES.copy()
            random.shuffle(ex_pool)
            ea, eb, ex_ans = _make_triplet(ex_pool[0], ex_pool[1], rule)
            examples.append((f"{ea} {eb}", ex_ans))

        return Stimulus(
            query=f"{a} {b}",
            expected=expected,
            few_shot_examples=examples,
            metadata={"rule": rule, "source": "generated"},
            answer_choices=[a, b],
        )

    def systematic_stimuli(self, n_per_rule: int, n_examples: int) -> list[Stimulus]:
        """Generate stimuli covering each rule with ``n_per_rule`` instances."""
        stimuli = []
        for rule in _RULES:
            for _ in range(n_per_rule):
                pool = _SYLLABLES.copy()
                random.shuffle(pool)
                a, b, expected = _make_triplet(pool[0], pool[1], rule)
                examples = []
                for _ in range(n_examples):
                    ex_pool = _SYLLABLES.copy()
                    random.shuffle(ex_pool)
                    ea, eb, ex_ans = _make_triplet(ex_pool[0], ex_pool[1], rule)
                    examples.append((f"{ea} {eb}", ex_ans))
                stimuli.append(
                    Stimulus(
                        query=f"{a} {b}",
                        expected=expected,
                        few_shot_examples=examples,
                        metadata={"rule": rule, "source": "systematic"},
                        answer_choices=[a, b],
                    )
                )
        return stimuli

    def format_completion(self, stimulus: Stimulus, choice: str) -> str:
        return " " + choice

    def score(self, response: ModelResponse, stimulus: Stimulus) -> bool:
        # first for an exact match
        if response.text.strip().lower() == stimulus.expected.strip().lower():
            return True
        else:
            # then split based on whitespace and check if the first part matches the expected
            return (
                response.text.strip().split()[0].lower()
                == stimulus.expected.strip().lower()
            )

    def build_prompt(self, stimulus: Stimulus, n_examples: int) -> str:
        if n_examples > 0 and stimulus.few_shot_examples:
            examples = stimulus.few_shot_examples[:n_examples]
            lines = [f"{q} {a}" for q, a in examples]
            lines.append(stimulus.query)
            return "\n".join(lines)
        return stimulus.query
