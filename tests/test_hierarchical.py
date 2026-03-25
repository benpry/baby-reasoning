import pytest
from baby_reasoning.tasks.base import ModelResponse, Stimulus
from baby_reasoning.tasks.hierarchical import HierarchicalTask


@pytest.fixture
def task():
    return HierarchicalTask()


def test_canonical_stimuli_loads(task):
    stimuli = task.canonical_stimuli()
    assert len(stimuli) >= 4
    for s in stimuli:
        assert s.expected in ("0", "1")


def test_canonical_stimuli_cover_both_answers(task):
    stimuli = task.canonical_stimuli()
    answers = {s.expected for s in stimuli}
    assert answers == {"0", "1"}


def test_canonical_stimuli_have_answer_choices(task):
    for s in task.canonical_stimuli():
        assert s.answer_choices == ["0", "1"]


def test_canonical_stimuli_pairs_have_no_spaces(task):
    for s in task.canonical_stimuli():
        assert " " not in s.query


def test_generate_stimulus_returns_valid_stimulus(task):
    s = task.generate_stimulus()
    assert isinstance(s, Stimulus)
    assert s.expected in ("0", "1")
    assert s.metadata.get("pattern") in (
        "same-same",
        "same-different",
        "different-different",
    )


def test_generate_stimulus_has_answer_choices(task):
    s = task.generate_stimulus()
    assert s.answer_choices == ["0", "1"]


def test_generate_stimulus_pairs_have_no_spaces(task):
    s = task.generate_stimulus()
    assert " " not in s.query
    for ex_query, _ in s.few_shot_examples:
        assert " " not in ex_query


def test_generate_stimulus_pairs_are_valid(task):
    """Each four-char string is two valid two-letter pairs."""
    for _ in range(10):
        s = task.generate_stimulus()
        # Query is exactly 4 uppercase letters
        assert len(s.query) == 4
        assert s.query.isalpha() and s.query.isupper()
        for ex_query, _ in s.few_shot_examples:
            assert len(ex_query) == 4
            assert ex_query.isalpha() and ex_query.isupper()


def test_generate_stimulus_n_examples_0(task):
    s = task.generate_stimulus(n_examples=0)
    assert s.few_shot_examples == []


def test_generate_stimulus_n_examples_5(task):
    s = task.generate_stimulus(n_examples=5)
    assert len(s.few_shot_examples) == 5


def test_systematic_stimuli_returns_correct_count(task):
    stimuli = task.systematic_stimuli(n_per_pattern=5, n_examples=3)
    assert len(stimuli) == 15  # 5 per pattern × 3 patterns


def test_systematic_stimuli_covers_all_patterns(task):
    stimuli = task.systematic_stimuli(n_per_pattern=5, n_examples=3)
    patterns = {s.metadata["pattern"] for s in stimuli}
    assert patterns == {"same-same", "same-different", "different-different"}


def test_systematic_stimuli_respects_n_examples(task):
    stimuli = task.systematic_stimuli(n_per_pattern=3, n_examples=5)
    for s in stimuli:
        assert len(s.few_shot_examples) == 5


def test_systematic_stimuli_many_examples(task):
    """Should handle large n_examples without running out of letters."""
    stimuli = task.systematic_stimuli(n_per_pattern=3, n_examples=9)
    for s in stimuli:
        assert len(s.few_shot_examples) == 9


def test_generate_stimulus_many_examples(task):
    """generate_stimulus should also handle large n_examples."""
    for _ in range(10):
        s = task.generate_stimulus(n_examples=9)
        assert len(s.few_shot_examples) == 9


def test_systematic_stimuli_n_examples_0(task):
    stimuli = task.systematic_stimuli(n_per_pattern=3, n_examples=0)
    for s in stimuli:
        assert s.few_shot_examples == []


def test_score_correct(task):
    s = Stimulus(query="AABB", expected="1")
    assert task.score(ModelResponse(text="1"), s) is True


def test_score_incorrect(task):
    s = Stimulus(query="AABB", expected="1")
    assert task.score(ModelResponse(text="0"), s) is False


def test_format_completion_prepends_space(task):
    s = Stimulus(query="AABB", expected="1")
    assert task.format_completion(s, "0") == " 0"


def test_build_prompt_zero_examples_is_raw_pair(task):
    s = Stimulus(
        query="AABB",
        expected="1",
        few_shot_examples=[("CCDD", "1"), ("EFGH", "0")],
    )
    prompt = task.build_prompt(s, n_examples=0)
    # Must contain the query pair
    assert "AABB" in prompt
    # Must NOT contain examples
    assert "CCDD" not in prompt
    # Must NOT contain any instruction text
    assert "same" not in prompt.lower()
    assert "different" not in prompt.lower()
    assert "judge" not in prompt.lower()


def test_build_prompt_with_examples_has_no_instructions(task):
    s = Stimulus(
        query="AABB",
        expected="1",
        few_shot_examples=[("CCDD", "1"), ("EFGH", "0")],
    )
    prompt = task.build_prompt(s, n_examples=2)
    assert "CCDD" in prompt
    assert "EFGH" in prompt
    assert "same" not in prompt.lower()
    assert "different" not in prompt.lower()
    assert "judge" not in prompt.lower()


def test_build_prompt_with_examples_format(task):
    s = Stimulus(
        query="AABB",
        expected="1",
        few_shot_examples=[("CCDD", "1"), ("EFGH", "0")],
    )
    prompt = task.build_prompt(s, n_examples=2)
    # Examples appear as "CCDD 1" and "EFGH 0", query at end
    assert "CCDD 1" in prompt
    assert "EFGH 0" in prompt
    assert prompt.rstrip().endswith("AABB")


def test_build_prompt_n_examples_1_shows_one_example(task):
    s = Stimulus(
        query="AABB",
        expected="1",
        few_shot_examples=[("CCDD", "1"), ("EFGH", "0")],
    )
    prompt = task.build_prompt(s, n_examples=1)
    assert "CCDD 1" in prompt
    assert "EFGH" not in prompt
    assert prompt.rstrip().endswith("AABB")
