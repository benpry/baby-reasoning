import pytest
from baby_reasoning.tasks.base import Condition, ModelResponse, Stimulus
from baby_reasoning.tasks.rules import RulesTask


@pytest.fixture
def task():
    return RulesTask()


def test_canonical_stimuli_loads(task):
    stimuli = task.canonical_stimuli()
    assert len(stimuli) >= 3
    for s in stimuli:
        assert isinstance(s, Stimulus)
        assert s.query
        assert s.expected


def test_canonical_stimuli_have_metadata(task):
    stimuli = task.canonical_stimuli()
    rules = {s.metadata["rule"] for s in stimuli}
    assert rules == {"ABA", "ABB", "AAB"}


def test_generate_stimulus_returns_valid_stimulus(task):
    s = task.generate_stimulus()
    assert isinstance(s, Stimulus)
    assert s.query
    assert s.expected
    assert s.metadata.get("rule") in ("ABA", "ABB", "AAB")


def test_score_correct(task):
    s = Stimulus(query="de ro ___", expected="ro", metadata={"rule": "ABB"})
    response = ModelResponse(text="ro")
    assert task.score(response, s) is True


def test_score_correct_with_extra_whitespace(task):
    s = Stimulus(query="de ro ___", expected="ro", metadata={"rule": "ABB"})
    response = ModelResponse(text="  ro  \n")
    assert task.score(response, s) is True


def test_score_incorrect(task):
    s = Stimulus(query="de ro ___", expected="ro", metadata={"rule": "ABB"})
    response = ModelResponse(text="de")
    assert task.score(response, s) is False


def test_build_prompt_zero_shot(task):
    s = Stimulus(
        query="de ro ___",
        expected="ro",
        few_shot_examples=[("ga ti ___", "ti")],
        metadata={"rule": "ABB"},
    )
    prompt = task.build_prompt(s, Condition.ZERO_SHOT)
    assert "de ro ___" in prompt
    # Zero-shot: no examples included
    assert "ga ti ___" not in prompt


def test_build_prompt_few_shot(task):
    s = Stimulus(
        query="de ro ___",
        expected="ro",
        few_shot_examples=[("ga ti ___", "ti"), ("li na ___", "na")],
        metadata={"rule": "ABB"},
    )
    prompt = task.build_prompt(s, Condition.FEW_SHOT)
    assert "de ro ___" in prompt
    assert "ga ti ___" in prompt
    assert "li na ___" in prompt
