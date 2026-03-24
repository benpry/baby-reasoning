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
        assert s.expected in ("same", "different")


def test_canonical_stimuli_cover_both_answers(task):
    stimuli = task.canonical_stimuli()
    answers = {s.expected for s in stimuli}
    assert answers == {"same", "different"}


def test_generate_stimulus_returns_valid_stimulus(task):
    s = task.generate_stimulus()
    assert isinstance(s, Stimulus)
    assert s.expected in ("same", "different")
    assert s.metadata.get("pattern") in (
        "same-same", "same-different", "different-different"
    )


def test_score_correct(task):
    s = Stimulus(query="AA BB", expected="same")
    response = ModelResponse(text="same")
    assert task.score(response, s) is True


def test_score_correct_case_insensitive(task):
    s = Stimulus(query="AA BB", expected="same")
    response = ModelResponse(text="Same")
    assert task.score(response, s) is True


def test_score_incorrect(task):
    s = Stimulus(query="AA BB", expected="same")
    response = ModelResponse(text="different")
    assert task.score(response, s) is False


def test_build_prompt_zero_shot(task):
    s = Stimulus(
        query="AA BB",
        expected="same",
        few_shot_examples=[("CC DD", "same"), ("GH IJ", "different")],
    )
    prompt = task.build_prompt(s, Condition.ZERO_SHOT)
    assert "AA BB" in prompt
    assert "same" in prompt or "different" in prompt
    # Zero-shot must not include the few-shot examples
    assert "CC DD" not in prompt
    assert "GH IJ" not in prompt


def test_build_prompt_few_shot_includes_examples(task):
    s = Stimulus(
        query="AA BB",
        expected="same",
        few_shot_examples=[("CC DD", "same"), ("GH IJ", "different")],
    )
    prompt = task.build_prompt(s, Condition.FEW_SHOT)
    assert "CC DD" in prompt
    assert "GH IJ" in prompt
