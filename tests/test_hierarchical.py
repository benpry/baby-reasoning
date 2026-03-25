import pytest
from baby_reasoning.tasks.base import Condition, ModelResponse, Stimulus
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


def test_generate_stimulus_letters_unique_across_pairs(task):
    s = task.generate_stimulus()
    all_pairs = [s.query] + [ex[0] for ex in s.few_shot_examples]
    seen = set()
    for pair in all_pairs:
        # A same-pair like "AA" uses one unique letter; different-pair uses two
        letters = set(pair)
        assert letters.isdisjoint(seen), (
            f"Letter overlap detected across pairs: {all_pairs}"
        )
        seen |= letters


def test_score_correct(task):
    s = Stimulus(query="AABB", expected="1")
    assert task.score(ModelResponse(text="1"), s) is True


def test_score_incorrect(task):
    s = Stimulus(query="AABB", expected="1")
    assert task.score(ModelResponse(text="0"), s) is False


def test_build_prompt_zero_shot_is_raw_pair(task):
    s = Stimulus(
        query="AABB",
        expected="1",
        few_shot_examples=[("CCDD", "1"), ("EFGH", "0")],
    )
    prompt = task.build_prompt(s, Condition.ZERO_SHOT)
    # Must contain the query pair
    assert "AABB" in prompt
    # Must NOT contain examples
    assert "CCDD" not in prompt
    # Must NOT contain any instruction text
    assert "same" not in prompt.lower()
    assert "different" not in prompt.lower()
    assert "judge" not in prompt.lower()


def test_build_prompt_few_shot_has_no_instructions(task):
    s = Stimulus(
        query="AABB",
        expected="1",
        few_shot_examples=[("CCDD", "1"), ("EFGH", "0")],
    )
    prompt = task.build_prompt(s, Condition.FEW_SHOT)
    assert "CCDD" in prompt
    assert "EFGH" in prompt
    assert "same" not in prompt.lower()
    assert "different" not in prompt.lower()
    assert "judge" not in prompt.lower()


def test_build_prompt_few_shot_format(task):
    s = Stimulus(
        query="AABB",
        expected="1",
        few_shot_examples=[("CCDD", "1"), ("EFGH", "0")],
    )
    prompt = task.build_prompt(s, Condition.FEW_SHOT)
    # Examples appear as "CCDD 1" and "EFGH 0", query at end
    assert "CCDD 1" in prompt
    assert "EFGH 0" in prompt
    assert prompt.rstrip().endswith("AABB")
