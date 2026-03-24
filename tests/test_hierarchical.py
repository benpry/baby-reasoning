import pytest
from baby_reasoning.tasks.base import Condition, ModelResponse, Stimulus
from baby_reasoning.tasks.hierarchical import HierarchicalTask


@pytest.fixture
def task():
    return HierarchicalTask()


# --- canonical_stimuli ---

def test_canonical_stimuli_loads(task):
    stimuli = task.canonical_stimuli()
    assert len(stimuli) >= 4
    for s in stimuli:
        assert isinstance(s, Stimulus)
        assert s.query
        assert s.expected


def test_canonical_stimuli_have_level_metadata(task):
    stimuli = task.canonical_stimuli()
    levels = {s.metadata["level"] for s in stimuli}
    # Must cover at least two levels (e.g. "same" and "different")
    assert len(levels) >= 2


def test_canonical_stimuli_expected_values_are_yes_or_no(task):
    stimuli = task.canonical_stimuli()
    for s in stimuli:
        assert s.expected.lower() in ("yes", "no")


# --- generate_stimulus ---

def test_generate_stimulus_returns_valid_stimulus(task):
    s = task.generate_stimulus()
    assert isinstance(s, Stimulus)
    assert s.query
    assert s.expected.lower() in ("yes", "no")
    assert "level" in s.metadata


def test_generate_stimulus_same_level_expects_yes(task):
    # Run enough times to hit both branches
    results = [task.generate_stimulus() for _ in range(30)]
    same = [s for s in results if s.metadata["level"] == "same"]
    assert all(s.expected.lower() == "yes" for s in same)


def test_generate_stimulus_different_level_expects_no(task):
    results = [task.generate_stimulus() for _ in range(30)]
    different = [s for s in results if s.metadata["level"] == "different"]
    assert all(s.expected.lower() == "no" for s in different)


# --- score ---

def test_score_correct_yes(task):
    s = Stimulus(query="Are A and B the same?", expected="yes")
    response = ModelResponse(text="yes")
    assert task.score(response, s) is True


def test_score_correct_no(task):
    s = Stimulus(query="Are A and B the same?", expected="no")
    response = ModelResponse(text="no")
    assert task.score(response, s) is True


def test_score_incorrect(task):
    s = Stimulus(query="Are A and B the same?", expected="yes")
    response = ModelResponse(text="no")
    assert task.score(response, s) is False


def test_score_case_insensitive(task):
    s = Stimulus(query="Are A and B the same?", expected="yes")
    response = ModelResponse(text="  YES  \n")
    assert task.score(response, s) is True


# --- build_prompt ---

def test_build_prompt_zero_shot_contains_query(task):
    s = Stimulus(
        query="ga ti | ga ro — same or different?",
        expected="no",
        few_shot_examples=[],
        metadata={"level": "different"},
    )
    prompt = task.build_prompt(s, Condition.ZERO_SHOT)
    assert s.query in prompt
    # Zero-shot: no examples
    assert "Examples:" not in prompt


def test_build_prompt_few_shot_includes_examples(task):
    s = Stimulus(
        query="ga ti | ga ro — same or different?",
        expected="no",
        few_shot_examples=[
            ("de de | ro ro — same or different?", "yes"),
            ("li na | li zo — same or different?", "no"),
        ],
        metadata={"level": "different"},
    )
    prompt = task.build_prompt(s, Condition.FEW_SHOT)
    assert s.query in prompt
    assert "de de | ro ro" in prompt
    assert "li na | li zo" in prompt


def test_build_prompt_asks_yes_no(task):
    s = Stimulus(
        query="ga ti | ga ro — same or different?",
        expected="no",
        few_shot_examples=[],
        metadata={"level": "different"},
    )
    prompt = task.build_prompt(s, Condition.ZERO_SHOT)
    lower = prompt.lower()
    assert "yes" in lower or "no" in lower
