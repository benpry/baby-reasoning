import pytest
import numpy as np

from baby_reasoning.tasks.base import ModelResponse, Stimulus
from baby_reasoning.tasks.matrix import (
    MatrixTask,
    _answer_is_empty,
    _format_answer,
    _format_cell,
    _prob_to_query,
)


@pytest.fixture
def task():
    return MatrixTask()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def test_format_cell_single_value():
    assert _format_cell(np.array([6])) == "6"


def test_format_cell_multi_value():
    assert _format_cell(np.array([0, 6])) == "0 6"


def test_format_cell_filters_negative_one():
    assert _format_cell(np.array([-1, 7])) == "7"


def test_format_cell_scalar_value():
    assert _format_cell(6) == "6"


def test_format_cell_scalar_negative_one():
    assert _format_cell(-1) == ""


def test_format_answer_multi_value():
    assert _format_answer(np.array([5, 2, 7])) == "5 2 7"


def test_format_answer_scalar():
    assert _format_answer(3) == "3"


def test_format_answer_filters_all_negatives():
    assert _format_answer(np.array([-1, -1])) == ""


def test_prob_to_query_formats_3x3_grid():
    # 3×3 array where each cell is a single integer
    prob = np.array([[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]])
    result = _prob_to_query(prob)
    rows = result.split("\n")
    assert len(rows) == 3
    assert rows[0] == "[1] [2] [3]"
    assert rows[1] == "[4] [5] [6]"
    # Bottom-right cell is open bracket only
    assert rows[2] == "[7] [8] ["


def test_answer_is_empty_empty_array():
    assert _answer_is_empty(np.array([], dtype=int)) is True


def test_answer_is_empty_all_sentinels():
    assert _answer_is_empty(np.array([-1, -1])) is True


def test_answer_is_empty_non_empty():
    assert _answer_is_empty(np.array([3])) is False


def test_generate_stimulus_never_returns_empty_expected(task):
    # AND_permuted has all-empty correct answers; generate_stimulus must never pick it
    for _ in range(20):
        s = task.generate_stimulus()
        assert len(s.expected) > 0, f"Got empty expected from rule_type={s.metadata['rule_type']}"


# ---------------------------------------------------------------------------
# canonical_stimuli — loads from npz
# ---------------------------------------------------------------------------

def test_canonical_stimuli_covers_all_rule_types(task):
    stimuli = task.canonical_stimuli()
    rule_types = {s.metadata["rule_type"] for s in stimuli}
    # AND_permuted has all-empty correct answers (empty-set intersections) and is
    # excluded from the canonical set since free generation cannot score empty responses.
    assert len(rule_types) == 31
    # Each type contributes up to _N_CANONICAL_PER_TYPE stimuli; expect at least 4 per type
    # (some types may have fewer non-empty problems, but not fewer than 4).
    assert len(stimuli) >= 31 * 4


def test_canonical_stimuli_have_valid_integer_expected(task):
    stimuli = task.canonical_stimuli()
    for s in stimuli:
        assert len(s.expected) > 0
        for token in s.expected.split():
            int(token)  # must parse as integer


def test_canonical_stimuli_queries_end_with_open_bracket(task):
    stimuli = task.canonical_stimuli()
    for s in stimuli:
        assert s.query.endswith("["), f"Query does not end with '[': {s.query!r}"


def test_canonical_stimuli_have_few_shot_examples(task):
    stimuli = task.canonical_stimuli()
    for s in stimuli:
        assert len(s.few_shot_examples) == 3


# ---------------------------------------------------------------------------
# generate_stimulus
# ---------------------------------------------------------------------------

def test_generate_stimulus_returns_valid_stimulus(task):
    s = task.generate_stimulus()
    assert isinstance(s, Stimulus)
    assert s.query.endswith("[")
    assert len(s.expected) > 0
    assert "rule_type" in s.metadata


def test_generate_stimulus_has_three_few_shot_examples(task):
    s = task.generate_stimulus()
    assert len(s.few_shot_examples) == 3
    for ex_query, ex_answer in s.few_shot_examples:
        assert ex_query.endswith("[")
        assert len(ex_answer) > 0


def test_generate_stimulus_n_examples_1(task):
    s = task.generate_stimulus(n_examples=1)
    assert len(s.few_shot_examples) == 1


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def test_score_exact_match(task):
    s = Stimulus(query="", expected="3", metadata={})
    assert task.score(ModelResponse(text="3"), s) is True


def test_score_strips_brackets(task):
    s = Stimulus(query="", expected="3", metadata={})
    assert task.score(ModelResponse(text="3]"), s) is True


def test_score_strips_trailing_bracket_multi_token(task):
    s = Stimulus(query="", expected="5 2 7", metadata={})
    assert task.score(ModelResponse(text="5 2 7]"), s) is True


def test_score_correct_with_continued_generation(task):
    """Model generates answer then keeps going: '3] [2] [2] [2]...'"""
    s = Stimulus(query="", expected="3", metadata={})
    assert task.score(ModelResponse(text="3] [2] [2] [2] [1] [27]"), s) is True


def test_score_perm_invariant_with_continued_generation(task):
    s = Stimulus(query="", expected="6 4 1 9", metadata={"perm_invariant": True})
    assert task.score(ModelResponse(text="4 6 9 1] [3] [2]"), s) is True


def test_score_does_not_match_response_with_leading_bracket(task):
    # Model should output "3]" (completing the open "["), not "[3]".
    # A leading "[" is not stripped; it causes a mismatch.
    s = Stimulus(query="", expected="3", metadata={})
    assert task.score(ModelResponse(text="[3]"), s) is False


def test_score_incorrect(task):
    s = Stimulus(query="", expected="3", metadata={})
    assert task.score(ModelResponse(text="5"), s) is False


def test_score_perm_invariant_order_independent(task):
    s = Stimulus(query="", expected="6 4 1 9", metadata={"perm_invariant": True})
    assert task.score(ModelResponse(text="4 6 9 1]"), s) is True


def test_score_perm_invariant_wrong_values(task):
    s = Stimulus(query="", expected="6 4 1 9", metadata={"perm_invariant": True})
    assert task.score(ModelResponse(text="6 4 1 8"), s) is False


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------

def test_build_prompt_zero_examples_is_raw_grid(task):
    s = Stimulus(
        query="[1] [2] [3]\n[4] [5] [6]\n[7] [8] [",
        expected="9",
        few_shot_examples=[("[0] [0] [0]\n[0] [0] [0]\n[0] [0] [", "0")],
        metadata={},
    )
    prompt = task.build_prompt(s, n_examples=0)
    assert prompt == s.query
    assert "[0]" not in prompt


def test_build_prompt_with_examples_prepends_completed_examples(task):
    s = Stimulus(
        query="[1] [2] [3]\n[4] [5] [6]\n[7] [8] [",
        expected="9",
        few_shot_examples=[("[3] [5] [7]\n[1] [3] [5]\n[5] [7] [", "9")],
        metadata={},
    )
    prompt = task.build_prompt(s, n_examples=1)
    assert "[3] [5] [7]" in prompt
    assert "[5] [7] [9]" in prompt  # example answer filled in
    assert "[7] [8] [" in prompt    # test query preserved
