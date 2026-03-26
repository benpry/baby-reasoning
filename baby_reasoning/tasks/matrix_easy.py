from __future__ import annotations

import json
import random

from baby_reasoning import DATA_DIR
from baby_reasoning.tasks.base import ModelResponse, Stimulus, Task

_DATA_PATH = DATA_DIR / "matrix_easy" / "tasks.json"
_DEFAULT_N_EXAMPLES = 3
_N_CANONICAL_PER_TYPE = 5
# Task types where the correct answer is a set (order-independent).
_PERM_INVARIANT_TYPES = {"logic_and"}


def _format_cell(cell) -> str:
    """Format a matrix cell as space-separated integers."""
    if isinstance(cell, list):
        return " ".join(str(v) for v in cell)
    return str(cell)


def _format_answer(option) -> str:
    """Format an answer option (scalar or list) as space-separated integers."""
    if isinstance(option, list):
        return " ".join(str(v) for v in option)
    return str(option)


def _matrix_to_query(matrix) -> str:
    """Convert a 3×3 matrix (with None at [2][2]) to a bracketed text grid.

    The bottom-right cell is omitted; the prompt ends with ``[`` so the model
    completes the missing cell.
    """
    rows = []
    for r in range(3):
        cells = []
        for c in range(3):
            if r == 2 and c == 2:
                cells.append("[")
            else:
                cells.append("[" + _format_cell(matrix[r][c]) + "]")
        rows.append(" ".join(cells))
    return "\n".join(rows)


def _load() -> dict[str, list]:
    """Load tasks.json and return tasks grouped by task_type."""
    with open(_DATA_PATH) as f:
        data = json.load(f)
    by_type: dict[str, list] = {}
    for task in data["tasks"]:
        by_type.setdefault(task["task_type"], []).append(task)
    return by_type


def _make_stimulus(task: dict, fs_pool: list[dict], n_examples: int) -> Stimulus:
    task_type = task["task_type"]
    perm_invariant = task_type in _PERM_INVARIANT_TYPES
    query = _matrix_to_query(task["matrix"])
    answer_choices = [_format_answer(opt) for opt in task["answer_options"]]
    expected = answer_choices[task["correct_index"]]
    examples = [
        (_matrix_to_query(ex["matrix"]), _format_answer(ex["answer_options"][ex["correct_index"]]))
        for ex in fs_pool[-n_examples:] if n_examples > 0
    ]
    return Stimulus(
        query=query,
        expected=expected,
        few_shot_examples=examples,
        metadata={"task_type": task_type, "perm_invariant": perm_invariant},
        answer_choices=answer_choices,
    )


class MatrixEasyTask(Task):
    """Easy digit matrix task drawn from the matrix_easy dataset.

    Problems are 3×3 grids with the same presentation format as MatrixTask
    (bracketed cells, bottom-right missing).  Each stimulus includes four
    matched answer options, enabling both free-generation and forced-choice
    log-prob scoring.

    Task types: constancy, pattern, pattern_tuple, progression, logic_and,
    logic_or_tuple.  ``logic_and`` problems are scored set-wise
    (perm_invariant=True); all others require exact token order.

    The last ``_DEFAULT_N_EXAMPLES`` problems of each type are reserved as the
    few-shot pool; the remainder form the test set.
    """

    def canonical_stimuli(self) -> list[Stimulus]:
        by_type = _load()
        stimuli = []
        for task_type, tasks in by_type.items():
            test_tasks = tasks[:-_DEFAULT_N_EXAMPLES]
            fs_pool = tasks[-_DEFAULT_N_EXAMPLES:]
            for task in test_tasks[:_N_CANONICAL_PER_TYPE]:
                stimuli.append(_make_stimulus(task, fs_pool, _DEFAULT_N_EXAMPLES))
        return stimuli

    def generate_stimulus(self, n_examples: int = _DEFAULT_N_EXAMPLES) -> Stimulus:
        by_type = _load()
        task_type = random.choice(list(by_type.keys()))
        tasks = by_type[task_type]
        test_tasks = tasks[:-_DEFAULT_N_EXAMPLES]
        fs_pool = tasks[-_DEFAULT_N_EXAMPLES:]
        task = random.choice(test_tasks)
        return _make_stimulus(task, fs_pool, n_examples)

    def score(self, response: ModelResponse, stimulus: Stimulus) -> bool:
        text = response.text.split("]")[0].strip()
        expected = stimulus.expected.strip()
        if stimulus.metadata.get("perm_invariant", False):
            return set(text.split()) == set(expected.split())
        return text == expected

    def format_completion(self, stimulus: Stimulus, choice: str) -> str:
        """Append the closing bracket so scored completions form a full cell."""
        return choice + "]"

    def build_prompt(self, stimulus: Stimulus, n_examples: int) -> str:
        if n_examples > 0 and stimulus.few_shot_examples:
            examples = stimulus.few_shot_examples[:n_examples]
            parts = []
            for ex_query, ex_answer in examples:
                parts.append(ex_query + ex_answer + "]")
                parts.append("")
            parts.append(stimulus.query)
            return "\n".join(parts)
        return stimulus.query
