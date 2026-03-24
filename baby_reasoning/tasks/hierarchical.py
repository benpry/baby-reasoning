from __future__ import annotations
import json
import random
from pathlib import Path

from baby_reasoning import DATA_DIR
from baby_reasoning.tasks.base import Condition, ModelResponse, Stimulus, Task

_SYLLABLES = [
    "ga", "ti", "li", "na", "ta", "da", "wo", "fe", "de", "ro",
    "ba", "fo", "bi", "ku", "me", "si", "pe", "zo", "re", "vi",
]

_LEVELS = ("same", "different")


def _make_pair(a: str, b: str, c: str, d: str, level: str) -> tuple[str, str]:
    """Return (query, expected) for a hierarchical equality trial.

    level='same'      — both pairs share the same within-pair relation
    level='different' — the two pairs have different within-pair relations

    A pair is either AA (both elements identical) or AB (elements differ).
    'same' means: AA|AA or AB|AB  →  expected 'yes'
    'different' means: AA|AB or AB|AA  →  expected 'no'
    """
    if level == "same":
        # Both pairs have the same relation type: use the supplied syllables
        # as-is; the caller arranges them so both pairs share the relation.
        query = f"{a} {b} | {c} {d} — same pattern?"
        return query, "yes"
    else:
        query = f"{a} {b} | {c} {d} — same pattern?"
        return query, "no"


def _sample_same_pair() -> tuple[str, str, str, str]:
    """Return four syllables (a,b,c,d) that form a 'same' trial.

    Either both pairs are AA-type or both are AB-type.
    """
    pool = _SYLLABLES.copy()
    random.shuffle(pool)
    pair_type = random.choice(("AA", "AB"))
    if pair_type == "AA":
        a = pool[0]
        c = pool[1]
        return a, a, c, c
    else:
        a, b, c, d = pool[0], pool[1], pool[2], pool[3]
        return a, b, c, d


def _sample_different_pair() -> tuple[str, str, str, str]:
    """Return four syllables (a,b,c,d) that form a 'different' trial.

    One pair is AA-type and the other is AB-type.
    """
    pool = _SYLLABLES.copy()
    random.shuffle(pool)
    if random.random() < 0.5:
        # first pair AA, second pair AB
        a = pool[0]
        c, d = pool[1], pool[2]
        return a, a, c, d
    else:
        # first pair AB, second pair AA
        a, b = pool[0], pool[1]
        c = pool[2]
        return a, b, c, c


class HierarchicalTask(Task):
    """Hierarchical equality task: judge whether two pairs share the same
    within-pair relation (both identical or both different)."""

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
        level = random.choice(_LEVELS)

        if level == "same":
            a, b, c, d = _sample_same_pair()
        else:
            a, b, c, d = _sample_different_pair()

        query, expected = _make_pair(a, b, c, d, level)

        # Build three few-shot examples with the same level
        examples = []
        for _ in range(3):
            if level == "same":
                ea, eb, ec, ed = _sample_same_pair()
            else:
                ea, eb, ec, ed = _sample_different_pair()
            ex_query, ex_answer = _make_pair(ea, eb, ec, ed, level)
            examples.append((ex_query, ex_answer))

        return Stimulus(
            query=query,
            expected=expected,
            few_shot_examples=examples,
            metadata={"level": level, "source": "generated"},
        )

    def score(self, response: ModelResponse, stimulus: Stimulus) -> bool:
        return response.text.strip().lower() == stimulus.expected.strip().lower()

    def build_prompt(self, stimulus: Stimulus, condition: Condition) -> str:
        lines = [
            "Two pairs of syllables are shown. Decide whether both pairs share the",
            "same internal pattern (both identical OR both different).",
            "Answer with only 'yes' or 'no'.",
            "",
        ]
        if condition == Condition.FEW_SHOT:
            lines.append("Examples:")
            for query, answer in stimulus.few_shot_examples:
                lines.append(f"  {query} → {answer}")
            lines.append("")
        lines.append(f"Now judge: {stimulus.query}")
        lines.append("Answer (yes/no):")
        return "\n".join(lines)
