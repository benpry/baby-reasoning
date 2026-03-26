import pytest
from baby_reasoning.tasks.base import ModelResponse, Stimulus
from baby_reasoning.tasks.rules import RulesTask


@pytest.fixture
def task():
    return RulesTask()


def test_canonical_stimuli_loads(task):
    stimuli = task.canonical_stimuli()
    assert len(stimuli) >= 2
    for s in stimuli:
        assert isinstance(s, Stimulus)
        assert s.query
        assert s.expected


def test_canonical_stimuli_have_metadata(task):
    stimuli = task.canonical_stimuli()
    rules = {s.metadata["rule"] for s in stimuli}
    # AAB dropped: only the two rules from Marcus 1999
    assert rules == {"ABA", "ABB"}


def test_canonical_stimuli_no_underscores(task):
    for s in task.canonical_stimuli():
        assert "___" not in s.query
        for ex_query, _ in s.few_shot_examples:
            assert "___" not in ex_query


def test_canonical_stimuli_have_answer_choices(task):
    for s in task.canonical_stimuli():
        assert s.answer_choices is not None
        assert len(s.answer_choices) == 2


def test_generate_stimulus_returns_valid_stimulus(task):
    s = task.generate_stimulus()
    assert isinstance(s, Stimulus)
    assert s.query
    assert s.expected
    assert s.metadata.get("rule") in ("ABA", "ABB")


def test_generate_stimulus_no_underscores(task):
    for _ in range(10):
        s = task.generate_stimulus()
        assert "___" not in s.query
        for ex_query, _ in s.few_shot_examples:
            assert "___" not in ex_query


def test_generate_stimulus_has_answer_choices(task):
    s = task.generate_stimulus()
    assert s.answer_choices is not None
    assert len(s.answer_choices) == 2
    # Answer choices are the two syllables in the query
    a, b = s.query.split()
    assert set(s.answer_choices) == {a, b}


def test_generate_stimulus_expected_is_one_of_query_syllables(task):
    for _ in range(10):
        s = task.generate_stimulus()
        a, b = s.query.split()
        assert s.expected in (a, b)


def test_generate_stimulus_n_examples_0(task):
    s = task.generate_stimulus(n_examples=0)
    assert s.few_shot_examples == []


def test_generate_stimulus_n_examples_5(task):
    s = task.generate_stimulus(n_examples=5)
    assert len(s.few_shot_examples) == 5


def test_generate_stimulus_n_examples_20(task):
    s = task.generate_stimulus(n_examples=20)
    assert len(s.few_shot_examples) == 20


def test_systematic_stimuli_n_examples_20(task):
    stimuli = task.systematic_stimuli(n_per_rule=3, n_examples=20)
    for s in stimuli:
        assert len(s.few_shot_examples) == 20


def test_systematic_stimuli_returns_correct_count(task):
    stimuli = task.systematic_stimuli(n_per_rule=5, n_examples=3)
    assert len(stimuli) == 10  # 5 per rule × 2 rules


def test_systematic_stimuli_covers_both_rules(task):
    stimuli = task.systematic_stimuli(n_per_rule=5, n_examples=3)
    rules = {s.metadata["rule"] for s in stimuli}
    assert rules == {"ABA", "ABB"}


def test_systematic_stimuli_respects_n_examples(task):
    stimuli = task.systematic_stimuli(n_per_rule=3, n_examples=5)
    for s in stimuli:
        assert len(s.few_shot_examples) == 5


def test_systematic_stimuli_n_examples_0(task):
    stimuli = task.systematic_stimuli(n_per_rule=3, n_examples=0)
    for s in stimuli:
        assert s.few_shot_examples == []


def test_score_correct(task):
    s = Stimulus(query="de ro", expected="ro", metadata={"rule": "ABB"})
    assert task.score(ModelResponse(text="ro"), s) is True


def test_score_correct_with_extra_whitespace(task):
    s = Stimulus(query="de ro", expected="ro", metadata={"rule": "ABB"})
    assert task.score(ModelResponse(text="  ro  \n"), s) is True


def test_score_incorrect(task):
    s = Stimulus(query="de ro", expected="ro", metadata={"rule": "ABB"})
    assert task.score(ModelResponse(text="de"), s) is False


def test_format_completion_prepends_space(task):
    s = Stimulus(query="de ro", expected="ro", metadata={"rule": "ABB"})
    assert task.format_completion(s, "ro") == " ro"


def test_build_prompt_zero_examples_is_two_syllables(task):
    s = Stimulus(
        query="de ro",
        expected="ro",
        few_shot_examples=[("ga ti", "ti")],
        metadata={"rule": "ABB"},
    )
    prompt = task.build_prompt(s, n_examples=0)
    assert "de ro" in prompt
    # Zero examples: no examples, no instruction text
    assert "ga ti" not in prompt
    assert "complete" not in prompt.lower()
    assert "fill" not in prompt.lower()


def test_build_prompt_with_examples_shows_complete_triplets(task):
    s = Stimulus(
        query="de ro",
        expected="ro",
        few_shot_examples=[("ga ti", "ti"), ("li na", "na")],
        metadata={"rule": "ABB"},
    )
    prompt = task.build_prompt(s, n_examples=2)
    # Examples appear as complete triplets: "ga ti ti", "li na na"
    assert "ga ti ti" in prompt
    assert "li na na" in prompt
    # Query at end (without answer)
    assert prompt.rstrip().endswith("de ro")
    # No instruction text
    assert "complete" not in prompt.lower()
    assert "fill" not in prompt.lower()
    assert "___" not in prompt


def test_build_prompt_n_examples_1_shows_one_example(task):
    s = Stimulus(
        query="de ro",
        expected="ro",
        few_shot_examples=[("ga ti", "ti"), ("li na", "na")],
        metadata={"rule": "ABB"},
    )
    prompt = task.build_prompt(s, n_examples=1)
    assert "ga ti ti" in prompt
    assert "li na na" not in prompt
    assert prompt.rstrip().endswith("de ro")
