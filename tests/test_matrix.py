import pytest
from baby_reasoning.tasks.base import Condition, ModelResponse, Stimulus
from baby_reasoning.tasks.matrix import MatrixTask


@pytest.fixture
def task():
    return MatrixTask()


def test_canonical_stimuli_loads(task):
    stimuli = task.canonical_stimuli()
    assert len(stimuli) >= 3
    for s in stimuli:
        assert s.expected
        assert "___" in s.query


def test_generate_stimulus_returns_valid_stimulus(task):
    s = task.generate_stimulus()
    assert isinstance(s, Stimulus)
    assert "___" in s.query
    assert s.expected
    assert s.metadata.get("rule")


def test_score_correct(task):
    s = Stimulus(query="Row 1: circle | triangle | square\nRow 3: square | circle | ___", expected="triangle")
    response = ModelResponse(text="triangle")
    assert task.score(response, s) is True


def test_score_correct_case_insensitive(task):
    s = Stimulus(query="...", expected="small square")
    response = ModelResponse(text="Small Square")
    assert task.score(response, s) is True


def test_score_incorrect(task):
    s = Stimulus(query="...", expected="triangle")
    response = ModelResponse(text="circle")
    assert task.score(response, s) is False


def test_build_prompt_zero_shot(task):
    s = Stimulus(
        query="Row 1: circle | triangle | square\nRow 3: square | circle | ___",
        expected="triangle",
    )
    prompt = task.build_prompt(s, Condition.ZERO_SHOT)
    assert "Row 1" in prompt
    assert "___" in prompt


def test_build_prompt_few_shot_includes_examples(task):
    s = Stimulus(
        query="Row 1: X | Y | Z\nRow 3: Z | X | ___",
        expected="Y",
        few_shot_examples=[
            ("Row 1: A | B | C\nRow 3: C | A | ___", "B"),
        ],
    )
    prompt = task.build_prompt(s, Condition.FEW_SHOT)
    assert "Row 1: A | B | C" in prompt
    assert "Row 1: X | Y | Z" in prompt


def test_build_prompt_zero_shot_excludes_examples(task):
    s = Stimulus(
        query="Row 1: X | Y | Z\nRow 3: Z | X | ___",
        expected="Y",
        few_shot_examples=[
            ("Row 1: A | B | C\nRow 3: C | A | ___", "B"),
        ],
    )
    prompt = task.build_prompt(s, Condition.ZERO_SHOT)
    assert "Row 1: A | B | C" not in prompt
