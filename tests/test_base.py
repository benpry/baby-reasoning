from dataclasses import fields
from baby_reasoning.tasks.base import (
    Condition, Stimulus, ModelResponse, TrialScore, TrialResult, Task, ModelBackend
)


def test_condition_values():
    assert Condition.ZERO_SHOT == "zero_shot"
    assert Condition.FEW_SHOT == "few_shot"


def test_stimulus_defaults():
    s = Stimulus(query="de ro ___", expected="ro")
    assert s.few_shot_examples == []
    assert s.metadata == {}


def test_stimulus_with_examples():
    s = Stimulus(
        query="de ro ___",
        expected="ro",
        few_shot_examples=[("ga ti ti", "ga ti ti")],
        metadata={"rule": "ABB"},
    )
    assert s.few_shot_examples == [("ga ti ti", "ga ti ti")]
    assert s.metadata["rule"] == "ABB"


def test_model_response_none_logprobs():
    r = ModelResponse(text="ro", token_logprobs=None)
    assert r.text == "ro"
    assert r.token_logprobs is None


def test_trial_score():
    ts = TrialScore(correct=True, logprob_correct=-1.2)
    assert ts.correct is True
    assert ts.logprob_correct == -1.2


def test_trial_score_none_logprob():
    ts = TrialScore(correct=False, logprob_correct=None)
    assert ts.logprob_correct is None


def test_task_is_abstract():
    import inspect
    assert inspect.isabstract(Task)


def test_model_backend_is_abstract():
    import inspect
    assert inspect.isabstract(ModelBackend)
