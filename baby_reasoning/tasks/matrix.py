from __future__ import annotations

import random

import numpy as np

from baby_reasoning import DATA_DIR
from baby_reasoning.tasks.base import Condition, ModelResponse, Stimulus, Task

# Number of problems reserved at the end of each type for few-shot examples.
_N_FEW_SHOT = 3
# Number of canonical test stimuli taken from the start of each type.
_N_CANONICAL_PER_TYPE = 5


def _format_cell(cell) -> str:
    """Format a cell (numpy array) as space-separated integers, filtering -1 sentinels."""
    if isinstance(cell, np.ndarray):
        values = [int(v) for v in cell.flat if int(v) != -1]
    else:
        val = int(cell)
        values = [val] if val != -1 else []
    return " ".join(str(v) for v in values)


def _prob_to_query(prob) -> str:
    """Convert a (3, 3, ...) problem array to a text grid.

    The bottom-right cell is omitted and the prompt ends with '[', following
    the presentation format of Webb et al. (2023): the model is expected to
    complete the open bracket with the missing cell's content.
    """
    rows = []
    for r in range(3):
        cells = []
        for c in range(3):
            if r == 2 and c == 2:
                cells.append("[")
            else:
                cells.append("[" + _format_cell(prob[r][c]) + "]")
        rows.append(" ".join(cells))
    return "\n".join(rows)


def _answer_is_empty(choice) -> bool:
    """Return True if the correct answer is the empty set (cannot test via free generation)."""
    if isinstance(choice, np.ndarray):
        return len([v for v in choice.flat if int(v) != -1]) == 0
    return False


def _format_answer(choice) -> str:
    """Format an answer choice as space-separated integers, filtering -1 sentinels."""
    if isinstance(choice, np.ndarray):
        values = [int(v) for v in choice.flat if int(v) != -1]
    else:
        val = int(choice)
        values = [val] if val != -1 else []
    return " ".join(str(v) for v in values)


class MatrixTask(Task):
    """Digit Matrices task from Webb et al. (2023).

    Problems are 3×3 grids of digit tokens governed by the same rule structure
    as Raven's Standard Progressive Matrices.  Each cell is presented as a
    bracketed sequence, e.g. ``[5 9 3]``.  The bottom-right cell is missing;
    the prompt ends with an open bracket ``[`` and the model is expected to
    generate the cell content followed by ``]``.

    Rule types (31 total):
    - Transformation: constant (row/col), distribution-of-3 (2 diagonals),
      progression (+1 / +2), and all 2- and 3-rule combinations thereof.
    - Logic: OR (set union, 3 column variants), AND, XOR, and spatially
      permuted versions of each.

    Scoring:
    - Transformation problems: exact digit order required.
    - Logic problems: set-based matching (``perm_invariant=True`` in the npz).
    """

    _DATA_PATH = DATA_DIR / "matrix" / "all_problems.npz"

    def _load(self) -> dict:
        d = np.load(self._DATA_PATH, allow_pickle=True)
        return d["all_problems"].item()

    def _make_stimulus(
        self,
        prob,
        answer_choices,
        correct_ind: int,
        rule_type: str,
        perm_invariant: bool,
        fs_data: dict,
    ) -> Stimulus:
        n = len(fs_data["prob"])
        fs_indices = range(n - _N_FEW_SHOT, n)
        query = _prob_to_query(prob)
        expected = _format_answer(answer_choices[correct_ind])
        examples = [
            (
                _prob_to_query(fs_data["prob"][i]),
                _format_answer(fs_data["answer_choices"][i][int(fs_data["correct_ind"][i])]),
            )
            for i in fs_indices
        ]
        return Stimulus(
            query=query,
            expected=expected,
            few_shot_examples=examples,
            metadata={"rule_type": rule_type, "perm_invariant": bool(perm_invariant)},
        )

    def canonical_stimuli(self) -> list[Stimulus]:
        all_problems = self._load()
        stimuli = []
        for rule_type, data in all_problems.items():
            n_test = len(data["prob"]) - _N_FEW_SHOT
            added = 0
            for i in range(n_test):
                if added >= _N_CANONICAL_PER_TYPE:
                    break
                correct = data["answer_choices"][i][int(data["correct_ind"][i])]
                if _answer_is_empty(correct):
                    continue
                stimuli.append(
                    self._make_stimulus(
                        data["prob"][i],
                        data["answer_choices"][i],
                        int(data["correct_ind"][i]),
                        rule_type,
                        data["perm_invariant"],
                        data,
                    )
                )
                added += 1
        return stimuli

    def generate_stimulus(self) -> Stimulus:
        all_problems = self._load()
        # Exclude types where every problem in the test range has an empty correct answer
        # (e.g. AND_permuted: all 100 correct answers are the empty intersection set).
        valid_types = [
            rt for rt, d in all_problems.items()
            if any(
                not _answer_is_empty(d["answer_choices"][i][int(d["correct_ind"][i])])
                for i in range(len(d["prob"]) - _N_FEW_SHOT)
            )
        ]
        rule_type = random.choice(valid_types)
        data = all_problems[rule_type]
        n_test = len(data["prob"]) - _N_FEW_SHOT
        # Sample until we find a problem with a non-empty correct answer
        indices = list(range(n_test))
        random.shuffle(indices)
        for idx in indices:
            correct = data["answer_choices"][idx][int(data["correct_ind"][idx])]
            if not _answer_is_empty(correct):
                break
        return self._make_stimulus(
            data["prob"][idx],
            data["answer_choices"][idx],
            int(data["correct_ind"][idx]),
            rule_type,
            data["perm_invariant"],
            data,
        )

    def score(self, response: ModelResponse, stimulus: Stimulus) -> bool:
        # Strip trailing ] that the model may include when completing [content]
        text = response.text.strip().rstrip("]").strip()
        expected = stimulus.expected.strip()
        if stimulus.metadata.get("perm_invariant", False):
            return set(text.split()) == set(expected.split())
        return text == expected

    def build_prompt(self, stimulus: Stimulus, condition: Condition) -> str:
        """Return the prompt for this stimulus.

        Zero-shot: the raw grid ending with ``[`` (following Webb et al. 2023).
        Few-shot: completed example grids (answer filled in) prepended, then
        the test grid ending with ``[``.
        """
        if condition == Condition.FEW_SHOT and stimulus.few_shot_examples:
            parts = []
            for ex_query, ex_answer in stimulus.few_shot_examples:
                # Fill in the answer to create a complete grid as context
                parts.append(ex_query + ex_answer + "]")
                parts.append("")
            parts.append(stimulus.query)
            return "\n".join(parts)
        return stimulus.query
