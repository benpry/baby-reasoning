import json
import dataclasses
from pathlib import Path
import pytest
from baby_reasoning.tasks.base import (
    Condition, ModelBackend, ModelResponse, Stimulus, Task, TrialResult,
)
from baby_reasoning.runner import evaluate, save_results


# --- Stubs ---

class StubBackend(ModelBackend):
    def __init__(self, text: str = "ro", logprob: float | None = -1.5):
        self._text = text
        self._logprob = logprob

    def generate(self, prompt: str) -> ModelResponse:
        return ModelResponse(text=self._text)

    def score_completion(self, prompt: str, completion: str) -> float | None:
        return self._logprob


class StubTask(Task):
    def canonical_stimuli(self) -> list[Stimulus]:
        return [
            Stimulus(query="de ro ___", expected="ro", metadata={"rule": "ABB"}),
            Stimulus(query="de ro ___", expected="de", metadata={"rule": "ABA"}),
        ]

    def generate_stimulus(self) -> Stimulus:
        return Stimulus(query="x y ___", expected="y")

    def score(self, response: ModelResponse, stimulus: Stimulus) -> bool:
        return response.text.strip().lower() == stimulus.expected.strip().lower()

    def build_prompt(self, stimulus: Stimulus, condition: Condition) -> str:
        return f"prompt: {stimulus.query}"


# --- Tests ---

def test_evaluate_returns_one_result_per_stimulus():
    results = evaluate(StubTask(), StubBackend(), Condition.ZERO_SHOT)
    assert len(results) == 2


def test_evaluate_correct_when_response_matches():
    results = evaluate(StubTask(), StubBackend(text="ro"), Condition.ZERO_SHOT)
    assert results[0].score.correct is True


def test_evaluate_incorrect_when_response_mismatches():
    results = evaluate(StubTask(), StubBackend(text="ro"), Condition.ZERO_SHOT)
    assert results[1].score.correct is False


def test_evaluate_populates_logprob_correct():
    results = evaluate(StubTask(), StubBackend(logprob=-2.5), Condition.ZERO_SHOT)
    assert results[0].score.logprob_correct == pytest.approx(-2.5)


def test_evaluate_handles_none_logprob():
    results = evaluate(StubTask(), StubBackend(logprob=None), Condition.ZERO_SHOT)
    assert results[0].score.logprob_correct is None


def test_evaluate_uses_custom_stimuli():
    custom = [Stimulus(query="x y ___", expected="x")]
    results = evaluate(StubTask(), StubBackend(text="x"), Condition.ZERO_SHOT, stimuli=custom)
    assert len(results) == 1
    assert results[0].score.correct is True


def test_evaluate_result_fields_populated():
    results = evaluate(StubTask(), StubBackend(), Condition.FEW_SHOT)
    r = results[0]
    assert r.task == "stub"
    assert r.condition == Condition.FEW_SHOT
    assert r.timestamp


def test_save_results_writes_json(tmp_path):
    results = evaluate(StubTask(), StubBackend(), Condition.ZERO_SHOT)
    path = save_results(results, model="test_model", task="stub", condition=Condition.ZERO_SHOT, results_dir=tmp_path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert isinstance(data, list)
    assert len(data) == 2


def test_save_results_path_structure(tmp_path):
    results = evaluate(StubTask(), StubBackend(), Condition.ZERO_SHOT)
    path = save_results(results, model="llama3:8b", task="rules", condition=Condition.ZERO_SHOT, results_dir=tmp_path)
    assert "llama3_8b" in str(path)
    assert "rules" in str(path)
    assert "zero_shot" in str(path)
