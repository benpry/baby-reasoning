import pytest

from baby_reasoning.tasks.base import ModelResponse, Stimulus
from baby_reasoning.tasks.matrix_easy import (
    MatrixEasyTask,
    _format_cell,
    _format_answer,
    _matrix_to_query,
)


@pytest.fixture
def task():
    return MatrixEasyTask()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def test_format_cell_scalar():
    assert _format_cell(5) == "5"


def test_format_cell_list_single():
    assert _format_cell([5]) == "5"


def test_format_cell_list_multi():
    assert _format_cell([5, 9]) == "5 9"


def test_format_answer_scalar():
    assert _format_answer(3) == "3"


def test_format_answer_list():
    assert _format_answer([4, 5]) == "4 5"


def test_matrix_to_query_scalar_cells():
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, None]]
    result = _matrix_to_query(matrix)
    rows = result.split("\n")
    assert rows[0] == "[1] [2] [3]"
    assert rows[1] == "[4] [5] [6]"
    assert rows[2] == "[7] [8] ["


def test_matrix_to_query_list_cells():
    matrix = [[[1, 2], [3], [4, 5]], [[6], [7, 8], [9]], [[1], [2], None]]
    result = _matrix_to_query(matrix)
    rows = result.split("\n")
    assert rows[0] == "[1 2] [3] [4 5]"
    assert rows[2].endswith("[")


# ---------------------------------------------------------------------------
# canonical_stimuli
# ---------------------------------------------------------------------------


def test_canonical_stimuli_covers_all_types(task):
    stimuli = task.canonical_stimuli()
    types = {s.metadata["task_type"] for s in stimuli}
    assert types == {"constancy", "pattern", "pattern_tuple", "progression", "logic_and", "logic_or_tuple"}


def test_canonical_stimuli_count(task):
    stimuli = task.canonical_stimuli()
    assert len(stimuli) == 30  # 5 per type × 6 types


def test_canonical_stimuli_queries_end_with_open_bracket(task):
    for s in task.canonical_stimuli():
        assert s.query.endswith("["), f"Expected '[' at end of query: {s.query!r}"


def test_canonical_stimuli_have_nonempty_expected(task):
    for s in task.canonical_stimuli():
        assert len(s.expected) > 0


def test_canonical_stimuli_have_four_answer_choices(task):
    for s in task.canonical_stimuli():
        assert s.answer_choices is not None
        assert len(s.answer_choices) == 4


def test_canonical_stimuli_expected_is_among_choices(task):
    for s in task.canonical_stimuli():
        assert s.expected in s.answer_choices, (
            f"expected={s.expected!r} not in choices={s.answer_choices}"
        )


def test_canonical_stimuli_have_three_few_shot_examples(task):
    for s in task.canonical_stimuli():
        assert len(s.few_shot_examples) == 3


# ---------------------------------------------------------------------------
# generate_stimulus
# ---------------------------------------------------------------------------


def test_generate_stimulus_returns_valid_stimulus(task):
    s = task.generate_stimulus()
    assert isinstance(s, Stimulus)
    assert s.query.endswith("[")
    assert len(s.expected) > 0
    assert "task_type" in s.metadata


def test_generate_stimulus_has_answer_choices(task):
    s = task.generate_stimulus()
    assert s.answer_choices is not None
    assert len(s.answer_choices) == 4
    assert s.expected in s.answer_choices


def test_generate_stimulus_n_examples_1(task):
    s = task.generate_stimulus(n_examples=1)
    assert len(s.few_shot_examples) == 1


def test_generate_stimulus_default_has_three_few_shot_examples(task):
    s = task.generate_stimulus()
    assert len(s.few_shot_examples) == 3


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def test_score_exact_match(task):
    s = Stimulus(query="", expected="5", metadata={})
    assert task.score(ModelResponse(text="5"), s) is True


def test_score_strips_closing_bracket(task):
    s = Stimulus(query="", expected="5", metadata={})
    assert task.score(ModelResponse(text="5]"), s) is True


def test_score_strips_trailing_content(task):
    s = Stimulus(query="", expected="5", metadata={})
    assert task.score(ModelResponse(text="5] more stuff"), s) is True


def test_score_incorrect(task):
    s = Stimulus(query="", expected="5", metadata={})
    assert task.score(ModelResponse(text="3"), s) is False


def test_score_perm_invariant(task):
    s = Stimulus(query="", expected="4 5", metadata={"perm_invariant": True})
    assert task.score(ModelResponse(text="5 4]"), s) is True


def test_score_perm_invariant_wrong_values(task):
    s = Stimulus(query="", expected="4 5", metadata={"perm_invariant": True})
    assert task.score(ModelResponse(text="4 6]"), s) is False


# ---------------------------------------------------------------------------
# format_completion
# ---------------------------------------------------------------------------


def test_format_completion_appends_closing_bracket(task):
    s = Stimulus(query="", expected="5", metadata={})
    assert task.format_completion(s, "5") == "5]"


def test_format_completion_multi_token(task):
    s = Stimulus(query="", expected="4 5", metadata={})
    assert task.format_completion(s, "4 5") == "4 5]"


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


def test_build_prompt_zero_examples_returns_raw_query(task):
    s = Stimulus(
        query="[1] [2] [3]\n[4] [5] [6]\n[7] [8] [",
        expected="9",
        few_shot_examples=[("[0] [0] [0]\n[0] [0] [0]\n[0] [0] [", "0")],
        metadata={},
    )
    assert task.build_prompt(s, n_examples=0) == s.query


def test_build_prompt_with_examples_fills_answer(task):
    s = Stimulus(
        query="[1] [2] [3]\n[4] [5] [6]\n[7] [8] [",
        expected="9",
        few_shot_examples=[("[3] [5] [7]\n[1] [3] [5]\n[5] [7] [", "9")],
        metadata={},
    )
    prompt = task.build_prompt(s, n_examples=1)
    assert "[5] [7] [9]" in prompt
    assert "[7] [8] [" in prompt


# ---------------------------------------------------------------------------
# perm_invariant metadata
# ---------------------------------------------------------------------------


def test_logic_and_stimuli_are_perm_invariant(task):
    stimuli = task.canonical_stimuli()
    logic_and = [s for s in stimuli if s.metadata["task_type"] == "logic_and"]
    assert all(s.metadata["perm_invariant"] for s in logic_and)


def test_non_logic_stimuli_are_not_perm_invariant(task):
    stimuli = task.canonical_stimuli()
    constancy = [s for s in stimuli if s.metadata["task_type"] == "constancy"]
    assert all(not s.metadata["perm_invariant"] for s in constancy)
