# LLM Evaluation Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular Python framework for evaluating small LLMs (via Ollama) on three developmental psychology tasks: Marcus rule learning, hierarchical equality, and text-based matrix reasoning.

**Architecture:** OOP with abstract base classes `Task` and `ModelBackend`; a standalone `evaluate()` runner assembles `TrialResult` dataclasses from task scoring and backend log probs; results persist as flat JSON files; reporting via Jupyter notebook.

**Tech Stack:** Python 3.11+, requests (Ollama HTTP), pytest, pandas, matplotlib, seaborn, nbformat, dataclasses

---

## File Map

| File                                   | Responsibility                                                                                        |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `pyproject.toml`                       | Package metadata, dependencies, pytest config                                                         |
| `baby_reasoning/__init__.py`           | `DATA_DIR`, `RESULTS_DIR` path constants                                                              |
| `baby_reasoning/tasks/base.py`         | `Condition`, `Stimulus`, `ModelResponse`, `TrialScore`, `TrialResult`, `Task` ABC, `ModelBackend` ABC |
| `baby_reasoning/tasks/rules.py`        | Marcus rule learning task                                                                             |
| `baby_reasoning/tasks/hierarchical.py` | Hierarchical equality task                                                                            |
| `baby_reasoning/tasks/matrix.py`       | Text-based matrix reasoning task                                                                      |
| `baby_reasoning/model.py`              | `OllamaBackend`                                                                                       |
| `baby_reasoning/runner.py`             | `evaluate()`, `save_results()`                                                                        |
| `data/rules/canonical.json`            | Canonical stimulus set for rules task                                                                 |
| `data/hierarchical/canonical.json`     | Canonical stimulus set for hierarchical equality                                                      |
| `data/matrix/canonical.json`           | Canonical stimulus set for matrix reasoning                                                           |
| `tests/test_base.py`                   | Dataclass and ABC unit tests                                                                          |
| `tests/test_rules.py`                  | Rules task unit tests                                                                                 |
| `tests/test_hierarchical.py`           | Hierarchical equality unit tests                                                                      |
| `tests/test_matrix.py`                 | Matrix reasoning unit tests                                                                           |
| `tests/test_model.py`                  | OllamaBackend with mock HTTP                                                                          |
| `tests/test_runner.py`                 | Runner unit tests with stub backend                                                                   |
| `notebooks/analysis.ipynb`             | Analysis notebook                                                                                     |

---

## Task 1: Project Scaffolding

**Files:**

- Create: `pyproject.toml`
- Create: `baby_reasoning/__init__.py`
- Create: `baby_reasoning/tasks/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "baby-reasoning"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "requests>=2.31",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-mock>=3.12",
    "responses>=0.25",
]
notebook = [
    "pandas>=2",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "nbformat>=5.9",
    "jupyter>=1.0",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["baby_reasoning*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package init with path constants**

```python
# baby_reasoning/__init__.py
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
```

- [ ] **Step 3: Create tasks sub-package init**

```python
# baby_reasoning/tasks/__init__.py
```

- [ ] **Step 4: Create data directories**

```bash
mkdir -p data/rules data/hierarchical data/matrix results notebooks tests
touch tests/__init__.py
```

- [ ] **Step 5: Install in editable mode**

```bash
pip install -e ".[dev,notebook]"
```

Expected: package installs without errors.

- [ ] **Step 6: Run tests to confirm baseline passes**

```bash
pytest
```

Expected: "no tests ran" or 0 collected (exit 0).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml baby_reasoning/ data/ results/ notebooks/ tests/
git commit -m "feat: project scaffolding — package structure and dependencies"
```

---

## Task 2: Core Abstractions

**Files:**

- Create: `baby_reasoning/tasks/base.py`
- Create: `tests/test_base.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_base.py
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
```

- [ ] **Step 2: Run to confirm it fails**

```bash
pytest tests/test_base.py -v
```

Expected: `ModuleNotFoundError: No module named 'baby_reasoning.tasks.base'`

- [ ] **Step 3: Implement `tasks/base.py`**

```python
# baby_reasoning/tasks/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class Condition(str, Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"


@dataclass
class Stimulus:
    query: str
    expected: str
    few_shot_examples: list[tuple[str, str]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ModelResponse:
    text: str
    token_logprobs: list[float] | None


@dataclass
class TrialScore:
    correct: bool
    logprob_correct: float | None


@dataclass
class TrialResult:
    model: str
    task: str
    condition: Condition
    stimulus: Stimulus
    response: ModelResponse
    score: TrialScore
    timestamp: str


class Task(ABC):
    @abstractmethod
    def canonical_stimuli(self) -> list[Stimulus]: ...

    @abstractmethod
    def generate_stimulus(self) -> Stimulus: ...

    @abstractmethod
    def score(self, response: ModelResponse, stimulus: Stimulus) -> bool: ...

    @abstractmethod
    def build_prompt(self, stimulus: Stimulus, condition: Condition) -> str: ...


class ModelBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> ModelResponse: ...

    @abstractmethod
    def score_completion(self, prompt: str, completion: str) -> float | None: ...
```

- [ ] **Step 4: Run to confirm it passes**

```bash
pytest tests/test_base.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add baby_reasoning/tasks/base.py tests/test_base.py
git commit -m "feat: core abstractions — Condition, Stimulus, ModelResponse, TrialScore, TrialResult, Task and ModelBackend ABCs"
```

---

## Task 3: Rules Task

**Files:**

- Create: `data/rules/canonical.json`
- Create: `baby_reasoning/tasks/rules.py`
- Create: `tests/test_rules.py`

### Background

The Marcus (1999) rule-learning task uses nonsense syllable triplets following one of three abstract rules:

- **ABA**: first and third elements are identical (e.g. "ga ti ga")
- **ABB**: second and third elements are identical (e.g. "ga ti ti")
- **AAB**: first and second elements are identical (e.g. "ga ga ti")

For exact-match scoring, queries use fill-in-the-blank format:

- ABA: `"de ro ___"` → expected `"de"` (repeat A)
- ABB: `"de ro ___"` → expected `"ro"` (repeat B)
- AAB: `"de ___ fo"` → expected `"de"` (repeat A)

**Note:** The canonical set below contains a minimal structurally-valid sample. Expand and refine stimuli based on Marcus (1999) before running experiments.

- [ ] **Step 1: Create `data/rules/canonical.json`**

```json
[
  {
    "query": "de ro ___",
    "expected": "ro",
    "few_shot_examples": [
      ["ga ti ___", "ti"],
      ["li na ___", "na"],
      ["ta da ___", "da"]
    ],
    "metadata": { "rule": "ABB", "source": "Marcus 1999" }
  },
  {
    "query": "de ro ___",
    "expected": "de",
    "few_shot_examples": [
      ["ga ti ___", "ga"],
      ["li na ___", "li"],
      ["ta da ___", "ta"]
    ],
    "metadata": { "rule": "ABA", "source": "Marcus 1999" }
  },
  {
    "query": "de ___ fo",
    "expected": "de",
    "few_shot_examples": [
      ["ga ___ ti", "ga"],
      ["li ___ na", "li"],
      ["ta ___ da", "ta"]
    ],
    "metadata": { "rule": "AAB", "source": "Marcus 1999" }
  }
]
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_rules.py
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
    response = ModelResponse(text="ro", token_logprobs=None)
    assert task.score(response, s) is True


def test_score_correct_with_extra_whitespace(task):
    s = Stimulus(query="de ro ___", expected="ro", metadata={"rule": "ABB"})
    response = ModelResponse(text="  ro  \n", token_logprobs=None)
    assert task.score(response, s) is True


def test_score_incorrect(task):
    s = Stimulus(query="de ro ___", expected="ro", metadata={"rule": "ABB"})
    response = ModelResponse(text="de", token_logprobs=None)
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
```

- [ ] **Step 3: Run to confirm it fails**

```bash
pytest tests/test_rules.py -v
```

Expected: `ModuleNotFoundError: No module named 'baby_reasoning.tasks.rules'`

- [ ] **Step 4: Implement `tasks/rules.py`**

```python
# baby_reasoning/tasks/rules.py
from __future__ import annotations
import json
import random
from pathlib import Path

from baby_reasoning import DATA_DIR
from baby_reasoning.tasks.base import Condition, ModelResponse, Stimulus, Task

_SYLLABLES = [
    "ga", "ti", "li", "na", "ta", "da", "wo", "fe", "de", "ro",
    "ba", "fo", "bi", "ku", "me", "si", "pe", "zo", "re", "vi",
]

_RULES = ("ABA", "ABB", "AAB")


def _make_triplet(a: str, b: str, rule: str) -> tuple[str, str]:
    """Return (query_with_blank, expected) for a given rule."""
    if rule == "ABA":
        return f"{a} {b} ___", a
    elif rule == "ABB":
        return f"{a} {b} ___", b
    else:  # AAB
        return f"{a} ___ {b}", a


class RulesTask(Task):
    """Marcus (1999) abstract rule learning over nonsense syllable triplets."""

    _DATA_PATH = DATA_DIR / "rules" / "canonical.json"

    def canonical_stimuli(self) -> list[Stimulus]:
        with open(self._DATA_PATH) as f:
            items = json.load(f)
        return [
            Stimulus(
                query=item["query"],
                expected=item["expected"],
                few_shot_examples=[tuple(e) for e in item.get("few_shot_examples", [])],
                metadata=item.get("metadata", {}),
            )
            for item in items
        ]

    def generate_stimulus(self) -> Stimulus:
        rule = random.choice(_RULES)
        pool = _SYLLABLES.copy()
        random.shuffle(pool)
        a, b = pool[0], pool[1]
        example_syllables = pool[2:]

        query, expected = _make_triplet(a, b, rule)

        # Build three few-shot examples using other syllables
        examples = []
        for i in range(3):
            ea, eb = example_syllables[i * 2], example_syllables[i * 2 + 1]
            ex_query, ex_answer = _make_triplet(ea, eb, rule)
            examples.append((ex_query, ex_answer))

        return Stimulus(
            query=query,
            expected=expected,
            few_shot_examples=examples,
            metadata={"rule": rule, "source": "generated"},
        )

    def score(self, response: ModelResponse, stimulus: Stimulus) -> bool:
        return response.text.strip().lower() == stimulus.expected.strip().lower()

    def build_prompt(self, stimulus: Stimulus, condition: Condition) -> str:
        lines = [
            "Complete the sequence by filling in the blank with a single word.",
            "",
        ]
        if condition == Condition.FEW_SHOT:
            lines.append("Examples:")
            for query, answer in stimulus.few_shot_examples:
                lines.append(f"  {query} → {answer}")
            lines.append("")
        lines.append(f"Now complete: {stimulus.query}")
        lines.append("Answer with only the missing word:")
        return "\n".join(lines)
```

- [ ] **Step 5: Run to confirm it passes**

```bash
pytest tests/test_rules.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add data/rules/canonical.json baby_reasoning/tasks/rules.py tests/test_rules.py
git commit -m "feat: rules task — Marcus ABA/ABB/AAB rule learning with canonical stimuli and generator"
```

---

## Task 4: Hierarchical Equality Task

**Files:**

- Create: `data/hierarchical/canonical.json`
- Create: `baby_reasoning/tasks/hierarchical.py`
- Create: `tests/test_hierarchical.py`

### Background

The hierarchical equality task (Marcus ~2001; see Geiger 2022) tests whether a model can recognize abstract relational structure at two levels:

- **Level 1**: whether two items in a pair are the same or different (e.g. "AA" → same, "AB" → different)
- **Level 2 (the task)**: whether two pairs share the same Level-1 relationship

Example stimuli:

- "AA BB" → both pairs are (same, same) → meta-relation: `"same"`
- "AA AB" → pairs are (same, different) → meta-relation: `"different"`
- "CD EF" → both pairs are (different, different) → meta-relation: `"same"`

**Note:** Expand canonical stimuli from Marcus (~2001) and Geiger (2022) before running experiments. Items below are structurally valid placeholders.

- [ ] **Step 1: Create `data/hierarchical/canonical.json`**

```json
[
  {
    "query": "AA BB",
    "expected": "same",
    "few_shot_examples": [
      ["CC DD", "same"],
      ["EE FF", "same"],
      ["GH IJ", "different"]
    ],
    "metadata": { "pattern": "same-same", "source": "Marcus 2001" }
  },
  {
    "query": "AA AB",
    "expected": "different",
    "few_shot_examples": [
      ["CC DD", "same"],
      ["EE FF", "same"],
      ["GH IJ", "different"]
    ],
    "metadata": { "pattern": "same-different", "source": "Marcus 2001" }
  },
  {
    "query": "CD EF",
    "expected": "same",
    "few_shot_examples": [
      ["CC DD", "same"],
      ["EE FF", "same"],
      ["GH IJ", "different"]
    ],
    "metadata": { "pattern": "different-different", "source": "Marcus 2001" }
  },
  {
    "query": "CC CD",
    "expected": "different",
    "few_shot_examples": [
      ["CC DD", "same"],
      ["EE FF", "same"],
      ["GH IJ", "different"]
    ],
    "metadata": { "pattern": "same-different", "source": "Marcus 2001" }
  }
]
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_hierarchical.py
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
    response = ModelResponse(text="same", token_logprobs=None)
    assert task.score(response, s) is True


def test_score_correct_case_insensitive(task):
    s = Stimulus(query="AA BB", expected="same")
    response = ModelResponse(text="Same", token_logprobs=None)
    assert task.score(response, s) is True


def test_score_incorrect(task):
    s = Stimulus(query="AA BB", expected="same")
    response = ModelResponse(text="different", token_logprobs=None)
    assert task.score(response, s) is False


def test_build_prompt_zero_shot(task):
    s = Stimulus(query="AA BB", expected="same")
    prompt = task.build_prompt(s, Condition.ZERO_SHOT)
    assert "AA BB" in prompt
    assert "same" in prompt or "different" in prompt  # instructions mention the options


def test_build_prompt_few_shot_includes_examples(task):
    s = Stimulus(
        query="AA BB",
        expected="same",
        few_shot_examples=[("CC DD", "same"), ("GH IJ", "different")],
    )
    prompt = task.build_prompt(s, Condition.FEW_SHOT)
    assert "CC DD" in prompt
    assert "GH IJ" in prompt
```

- [ ] **Step 3: Run to confirm it fails**

```bash
pytest tests/test_hierarchical.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 4: Implement `tasks/hierarchical.py`**

```python
# baby_reasoning/tasks/hierarchical.py
from __future__ import annotations
import json
import random
from pathlib import Path

from baby_reasoning import DATA_DIR
from baby_reasoning.tasks.base import Condition, ModelResponse, Stimulus, Task

_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _pair_relation(pair: str) -> str:
    """'AA' → 'same', 'AB' → 'different'."""
    return "same" if pair[0] == pair[1] else "different"


def _random_pair(same: bool) -> str:
    pool = _LETTERS.copy()
    random.shuffle(pool)
    if same:
        return pool[0] * 2
    else:
        return pool[0] + pool[1]


class HierarchicalTask(Task):
    """Hierarchical equality task: judge whether two pairs share the same Level-1 relation."""

    _DATA_PATH = DATA_DIR / "hierarchical" / "canonical.json"

    def canonical_stimuli(self) -> list[Stimulus]:
        with open(self._DATA_PATH) as f:
            items = json.load(f)
        return [
            Stimulus(
                query=item["query"],
                expected=item["expected"],
                few_shot_examples=[tuple(e) for e in item.get("few_shot_examples", [])],
                metadata=item.get("metadata", {}),
            )
            for item in items
        ]

    def generate_stimulus(self) -> Stimulus:
        # Randomly choose a Level-2 pattern
        rel1 = random.choice([True, False])  # same or different for pair 1
        rel2 = random.choice([True, False])  # same or different for pair 2
        pair1 = _random_pair(rel1)
        pair2 = _random_pair(rel2)
        meta_rel = "same" if rel1 == rel2 else "different"

        r1_str = "same" if rel1 else "different"
        r2_str = "same" if rel2 else "different"
        pattern = f"{r1_str}-{r2_str}"

        # Build three few-shot examples covering diverse patterns
        examples = []
        for _ in range(3):
            e_rel1 = random.choice([True, False])
            e_rel2 = random.choice([True, False])
            e_pair1 = _random_pair(e_rel1)
            e_pair2 = _random_pair(e_rel2)
            e_answer = "same" if e_rel1 == e_rel2 else "different"
            examples.append((f"{e_pair1} {e_pair2}", e_answer))

        return Stimulus(
            query=f"{pair1} {pair2}",
            expected=meta_rel,
            few_shot_examples=examples,
            metadata={"pattern": pattern, "source": "generated"},
        )

    def score(self, response: ModelResponse, stimulus: Stimulus) -> bool:
        return response.text.strip().lower() == stimulus.expected.strip().lower()

    def build_prompt(self, stimulus: Stimulus, condition: Condition) -> str:
        lines = [
            "Two pairs of letters are shown. Each pair is either 'same' (both letters identical)",
            "or 'different' (letters differ). Judge whether the two pairs share the SAME",
            "first-level relationship, or have DIFFERENT first-level relationships.",
            "Answer with only 'same' or 'different'.",
            "",
        ]
        if condition == Condition.FEW_SHOT:
            lines.append("Examples:")
            for query, answer in stimulus.few_shot_examples:
                lines.append(f"  {query} → {answer}")
            lines.append("")
        lines.append(f"Pairs: {stimulus.query}")
        lines.append("Answer:")
        return "\n".join(lines)
```

- [ ] **Step 5: Run to confirm it passes**

```bash
pytest tests/test_hierarchical.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add data/hierarchical/canonical.json baby_reasoning/tasks/hierarchical.py tests/test_hierarchical.py
git commit -m "feat: hierarchical equality task with canonical stimuli and generator"
```

---

## Task 5: Matrix Reasoning Task

**Files:**

- Create: `data/matrix/canonical.json`
- Create: `baby_reasoning/tasks/matrix.py`
- Create: `tests/test_matrix.py`

### Background

Text-encoded Raven's Progressive Matrices (Webb et al. 2023). A 3×3 grid where items vary along one or more attributes (shape, size, color, etc.) following row- and column-wise rules. The bottom-right cell is missing; the model must generate it.

Example (single attribute, cycling rule):

```
Row 1: circle | triangle | square
Row 2: triangle | square | circle
Row 3: square | circle | ___
```

Expected: `"triangle"`

**Note:** Expand canonical stimuli from Webb et al. (2023) before running experiments. The items below are structurally valid examples.

- [ ] **Step 1: Create `data/matrix/canonical.json`**

```json
[
  {
    "query": "Row 1: circle | triangle | square\nRow 2: triangle | square | circle\nRow 3: square | circle | ___",
    "expected": "triangle",
    "few_shot_examples": [],
    "metadata": {
      "rule": "rotation",
      "attributes": ["shape"],
      "source": "Webb 2023"
    }
  },
  {
    "query": "Row 1: large circle | medium circle | small circle\nRow 2: large triangle | medium triangle | small triangle\nRow 3: large square | medium square | ___",
    "expected": "small square",
    "few_shot_examples": [],
    "metadata": {
      "rule": "size-progression",
      "attributes": ["size", "shape"],
      "source": "Webb 2023"
    }
  },
  {
    "query": "Row 1: red circle | blue circle | green circle\nRow 2: red triangle | blue triangle | green triangle\nRow 3: red square | blue square | ___",
    "expected": "green square",
    "few_shot_examples": [],
    "metadata": {
      "rule": "color-constant",
      "attributes": ["color", "shape"],
      "source": "Webb 2023"
    }
  }
]
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_matrix.py
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
    response = ModelResponse(text="triangle", token_logprobs=None)
    assert task.score(response, s) is True


def test_score_correct_case_insensitive(task):
    s = Stimulus(query="...", expected="small square")
    response = ModelResponse(text="Small Square", token_logprobs=None)
    assert task.score(response, s) is True


def test_score_incorrect(task):
    s = Stimulus(query="...", expected="triangle")
    response = ModelResponse(text="circle", token_logprobs=None)
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
```

- [ ] **Step 3: Run to confirm it fails**

```bash
pytest tests/test_matrix.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 4: Implement `tasks/matrix.py`**

```python
# baby_reasoning/tasks/matrix.py
from __future__ import annotations
import json
import random
from pathlib import Path

from baby_reasoning import DATA_DIR
from baby_reasoning.tasks.base import Condition, ModelResponse, Stimulus, Task

_SHAPES = ["circle", "triangle", "square", "pentagon", "hexagon"]
_SIZES = ["large", "medium", "small"]
_COLORS = ["red", "blue", "green", "yellow", "purple"]


def _rotation_matrix(items: list[str]) -> tuple[str, str]:
    """Generate a 3x3 rotation matrix query and expected answer."""
    a, b, c = items[0], items[1], items[2]
    rows = [
        f"Row 1: {a} | {b} | {c}",
        f"Row 2: {b} | {c} | {a}",
        f"Row 3: {c} | {a} | ___",
    ]
    return "\n".join(rows), b


class MatrixTask(Task):
    """Text-encoded Raven's Progressive Matrices (Webb et al. 2023)."""

    _DATA_PATH = DATA_DIR / "matrix" / "canonical.json"

    def canonical_stimuli(self) -> list[Stimulus]:
        with open(self._DATA_PATH) as f:
            items = json.load(f)
        return [
            Stimulus(
                query=item["query"],
                expected=item["expected"],
                few_shot_examples=[tuple(e) for e in item.get("few_shot_examples", [])],
                metadata=item.get("metadata", {}),
            )
            for item in items
        ]

    def generate_stimulus(self) -> Stimulus:
        shapes = random.sample(_SHAPES, 3)
        query, expected = _rotation_matrix(shapes)
        return Stimulus(
            query=query,
            expected=expected,
            few_shot_examples=[],
            metadata={"rule": "rotation", "attributes": ["shape"], "source": "generated"},
        )

    def score(self, response: ModelResponse, stimulus: Stimulus) -> bool:
        return response.text.strip().lower() == stimulus.expected.strip().lower()

    def build_prompt(self, stimulus: Stimulus, condition: Condition) -> str:
        lines = [
            "Complete the pattern matrix by filling in the missing cell (marked ___).",
            "Answer with only the missing item.",
            "",
        ]
        if condition == Condition.FEW_SHOT and stimulus.few_shot_examples:
            lines.append("Examples:")
            for query, answer in stimulus.few_shot_examples:
                lines.append(f"  Matrix:\n    {query.replace(chr(10), chr(10) + '    ')}")
                lines.append(f"  Answer: {answer}")
                lines.append("")
        lines.append("Matrix:")
        lines.append(stimulus.query)
        lines.append("Answer:")
        return "\n".join(lines)
```

- [ ] **Step 5: Run to confirm it passes**

```bash
pytest tests/test_matrix.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add data/matrix/canonical.json baby_reasoning/tasks/matrix.py tests/test_matrix.py
git commit -m "feat: matrix reasoning task with canonical stimuli and rotation generator"
```

---

## Task 6: Ollama Model Backend

**Files:**

- Create: `baby_reasoning/model.py`
- Create: `tests/test_model.py`

### Ollama API Notes

- Generate endpoint: `POST http://localhost:11434/api/generate`
- Request body: `{"model": "llama3:8b", "prompt": "...", "stream": false, "options": {"num_predict": 50}}`
- Response: `{"response": "...", "done": true, ...}`
- Log probs: pass `"logprobs": true` in the request; Ollama returns per-token log probs in `logprobs.token_logprobs` when supported. If not supported, the field is absent or null.
- For `score_completion`: send `prompt + completion` as the full prompt with `num_predict: 0` to score without generating. Parse the completion tokens' log probs from the response. Returns `None` if log probs are unavailable.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py
import pytest
import responses as resp
import json
from baby_reasoning.model import OllamaBackend
from baby_reasoning.tasks.base import ModelResponse


@pytest.fixture
def backend():
    return OllamaBackend(model="test-model", base_url="http://localhost:11434")


@resp.activate
def test_generate_returns_text(backend):
    resp.add(
        resp.POST,
        "http://localhost:11434/api/generate",
        json={"response": "ro", "done": True},
        status=200,
    )
    result = backend.generate("some prompt")
    assert isinstance(result, ModelResponse)
    assert result.text == "ro"


@resp.activate
def test_generate_strips_trailing_whitespace(backend):
    resp.add(
        resp.POST,
        "http://localhost:11434/api/generate",
        json={"response": "ro\n", "done": True},
        status=200,
    )
    result = backend.generate("some prompt")
    assert result.text == "ro"


@resp.activate
def test_generate_returns_none_logprobs_when_absent(backend):
    resp.add(
        resp.POST,
        "http://localhost:11434/api/generate",
        json={"response": "ro", "done": True},
        status=200,
    )
    result = backend.generate("some prompt")
    assert result.token_logprobs is None


@resp.activate
def test_generate_returns_logprobs_when_present(backend):
    resp.add(
        resp.POST,
        "http://localhost:11434/api/generate",
        json={
            "response": "ro",
            "done": True,
            "logprobs": {"token_logprobs": [-1.2, -0.5]},
        },
        status=200,
    )
    result = backend.generate("some prompt")
    assert result.token_logprobs == [-1.2, -0.5]


@resp.activate
def test_score_completion_returns_none_when_logprobs_absent(backend):
    resp.add(
        resp.POST,
        "http://localhost:11434/api/generate",
        json={"response": "", "done": True},
        status=200,
    )
    result = backend.score_completion("prompt", "ro")
    assert result is None


@resp.activate
def test_score_completion_sums_logprobs(backend):
    resp.add(
        resp.POST,
        "http://localhost:11434/api/generate",
        json={
            "response": "",
            "done": True,
            "logprobs": {"token_logprobs": [-1.0, -2.0, -0.5]},
        },
        status=200,
    )
    result = backend.score_completion("prompt ", "ro fe")
    assert result == pytest.approx(-3.5)


@resp.activate
def test_model_name_sent_in_request(backend):
    resp.add(
        resp.POST,
        "http://localhost:11434/api/generate",
        json={"response": "x", "done": True},
        status=200,
    )
    backend.generate("hi")
    request_body = json.loads(resp.calls[0].request.body)
    assert request_body["model"] == "test-model"
```

- [ ] **Step 2: Run to confirm it fails**

```bash
pytest tests/test_model.py -v
```

Expected: `ModuleNotFoundError: No module named 'baby_reasoning.model'`

- [ ] **Step 3: Implement `model.py`**

```python
# baby_reasoning/model.py
from __future__ import annotations
import requests
from baby_reasoning.tasks.base import ModelBackend, ModelResponse


class OllamaBackend(ModelBackend):
    """ModelBackend implementation over the Ollama HTTP API."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _post(self, payload: dict) -> dict:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def generate(self, prompt: str) -> ModelResponse:
        data = self._post({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "logprobs": True,
            "options": {"num_predict": 50},
        })
        logprobs_data = data.get("logprobs")
        token_logprobs = (
            logprobs_data.get("token_logprobs")
            if isinstance(logprobs_data, dict)
            else None
        )
        return ModelResponse(
            text=data.get("response", "").rstrip(),
            token_logprobs=token_logprobs,
        )

    def score_completion(self, prompt: str, completion: str) -> float | None:
        """Return sum of token log probs for the completion, or None if unsupported."""
        data = self._post({
            "model": self.model,
            "prompt": prompt + completion,
            "stream": False,
            "logprobs": True,
            "options": {"num_predict": 0},
        })
        logprobs_data = data.get("logprobs")
        if not isinstance(logprobs_data, dict):
            return None
        token_logprobs = logprobs_data.get("token_logprobs")
        if not token_logprobs:
            return None
        return sum(token_logprobs)
```

- [ ] **Step 4: Run to confirm it passes**

```bash
pytest tests/test_model.py -v
```

Expected: all 7 tests PASS.

> **Note on log prob implementation:** Ollama's support for `logprobs` varies by version and model. If `score_completion` always returns `None` in practice, investigate whether your Ollama version exposes this field. The framework handles `None` gracefully throughout, so experiments can proceed without log probs while this is investigated.

- [ ] **Step 5: Commit**

```bash
git add baby_reasoning/model.py tests/test_model.py
git commit -m "feat: OllamaBackend with generate() and score_completion() over Ollama HTTP API"
```

---

## Task 7: Runner and Result Serialization

**Files:**

- Create: `baby_reasoning/runner.py`
- Create: `tests/test_runner.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runner.py
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
        return ModelResponse(text=self._text, token_logprobs=None)

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
    # First stimulus expected "ro"
    assert results[0].score.correct is True


def test_evaluate_incorrect_when_response_mismatches():
    results = evaluate(StubTask(), StubBackend(text="ro"), Condition.ZERO_SHOT)
    # Second stimulus expected "de"
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
    # model tag colons replaced with underscores
    assert "llama3_8b" in str(path)
    assert "rules" in str(path)
    assert "zero_shot" in str(path)
```

- [ ] **Step 2: Run to confirm it fails**

```bash
pytest tests/test_runner.py -v
```

Expected: `ModuleNotFoundError: No module named 'baby_reasoning.runner'`

- [ ] **Step 3: Implement `runner.py`**

Note: `test_evaluate_result_fields_populated` expects `r.task == "stub"`. The runner derives the task name from the class name: `StubTask` → `"stub"`. Implement accordingly.

```python
# baby_reasoning/runner.py
from __future__ import annotations
import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

from baby_reasoning import RESULTS_DIR
from baby_reasoning.tasks.base import (
    Condition, ModelBackend, Stimulus, Task, TrialResult, TrialScore,
)


def _task_name(task: Task) -> str:
    """Derive snake-case task name from class name, e.g. RulesTask → 'rules'."""
    name = type(task).__name__
    name = name.removesuffix("Task")
    # CamelCase → snake_case
    import re
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def evaluate(
    task: Task,
    backend: ModelBackend,
    condition: Condition,
    stimuli: list[Stimulus] | None = None,
) -> list[TrialResult]:
    if stimuli is None:
        stimuli = task.canonical_stimuli()

    task_name = _task_name(task)
    results = []

    for stimulus in stimuli:
        prompt = task.build_prompt(stimulus, condition)
        response = backend.generate(prompt)
        correct = task.score(response, stimulus)
        logprob_correct = backend.score_completion(prompt, stimulus.expected)
        score = TrialScore(correct=correct, logprob_correct=logprob_correct)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        results.append(TrialResult(
            model=getattr(backend, "model", "unknown"),
            task=task_name,
            condition=condition,
            stimulus=stimulus,
            response=response,
            score=score,
            timestamp=timestamp,
        ))

    return results


def save_results(
    results: list[TrialResult],
    model: str,
    task: str,
    condition: Condition,
    results_dir: Path | None = None,
) -> Path:
    if results_dir is None:
        results_dir = RESULTS_DIR

    model_tag = model.replace(":", "_")
    # Use timestamp from first result for the filename (replace colons with hyphens)
    ts = results[0].timestamp.replace(":", "-") if results else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%fZ")

    out_dir = results_dir / model_tag / task / condition.value
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ts}.json"

    data = [dataclasses.asdict(r) for r in results]
    out_path.write_text(json.dumps(data, indent=2))
    return out_path
```

- [ ] **Step 4: Run to confirm it passes**

```bash
pytest tests/test_runner.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 5: Run the full test suite**

```bash
pytest -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add baby_reasoning/runner.py tests/test_runner.py
git commit -m "feat: evaluate() runner and save_results() serialization"
```

---

## Task 8: Analysis Notebook

**Files:**

- Create: `notebooks/analysis.ipynb`

- [ ] **Step 1: Create the analysis notebook**

```python
# Run this once to generate the notebook file:
# python -c "import create_notebook; create_notebook.run()"
# Or create it manually using the cell contents below.
```

Create `notebooks/analysis.ipynb` as a Jupyter notebook with these cells:

**Cell 1 — Imports:**

```python
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = Path("../results")
```

**Cell 2 — Load all results:**

```python
records = []
for json_file in RESULTS_DIR.rglob("*.json"):
    data = json.loads(json_file.read_text())
    for item in data:
        records.append({
            "model": item["model"],
            "task": item["task"],
            "condition": item["condition"],
            "query": item["stimulus"]["query"],
            "expected": item["stimulus"]["expected"],
            "few_shot_examples": item["stimulus"]["few_shot_examples"],
            "metadata": item["stimulus"]["metadata"],
            "response_text": item["response"]["text"],
            "token_logprobs": item["response"]["token_logprobs"],
            "correct": item["score"]["correct"],
            "logprob_correct": item["score"]["logprob_correct"],
            "timestamp": item["timestamp"],
        })

df = pd.DataFrame(records)
print(f"Loaded {len(df)} trials from {df['model'].nunique()} models")
df.head()
```

**Cell 3 — Accuracy table (model × task × condition):**

```python
accuracy = (
    df.groupby(["model", "task", "condition"])["correct"]
    .mean()
    .mul(100)
    .round(1)
    .unstack("condition")
    .reset_index()
)
accuracy.columns.name = None
print(accuracy.to_string(index=False))
```

**Cell 4 — Accuracy heatmap:**

```python
pivot = df.groupby(["model", "task"])["correct"].mean().unstack("task")
plt.figure(figsize=(8, max(3, len(pivot) * 0.6)))
sns.heatmap(pivot, annot=True, fmt=".0%", cmap="YlGn", vmin=0, vmax=1)
plt.title("Accuracy by model and task")
plt.tight_layout()
plt.show()
```

**Cell 5 — Log probability distributions:**

```python
lp_df = df.dropna(subset=["logprob_correct"])
if lp_df.empty:
    print("No log prob data available yet.")
else:
    g = sns.FacetGrid(lp_df, col="task", row="condition", height=3, aspect=1.5)
    g.map_dataframe(sns.boxplot, x="model", y="logprob_correct")
    g.set_axis_labels("Model", "Log P(correct)")
    g.set_titles("{row_name} | {col_name}")
    plt.tight_layout()
    plt.show()
```

**Cell 6 — Per-task breakdown by rule/pattern type:**

```python
df["rule"] = df["metadata"].apply(lambda m: m.get("rule") or m.get("pattern", "unknown"))
rule_acc = (
    df.groupby(["task", "rule", "condition"])["correct"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "accuracy", "count": "n"})
    .reset_index()
)
rule_acc["accuracy"] = rule_acc["accuracy"].mul(100).round(1)
print(rule_acc.to_string(index=False))
```

To create this as an `.ipynb` file, run:

```bash
pip install nbformat
python - <<'EOF'
import nbformat

cells = [
    ("import json\nfrom pathlib import Path\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nRESULTS_DIR = Path('../results')", "code"),
    # ... (paste each cell source above)
]

nb = nbformat.v4.new_notebook()
nb.cells = [nbformat.v4.new_code_cell(src) for src, _ in cells]
with open("notebooks/analysis.ipynb", "w") as f:
    nbformat.write(nb, f)
print("Created notebooks/analysis.ipynb")
EOF
```

Or create the `.ipynb` manually in Jupyter and paste each cell's code.

- [ ] **Step 2: Verify the notebook loads without errors (no results yet)**

```bash
cd notebooks && jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=60 analysis.ipynb --output analysis_test.ipynb 2>&1 | head -20
rm -f analysis_test.ipynb
```

Expected: executes without Python errors (Cell 2 will show 0 trials loaded; Cell 5 will print "No log prob data available yet.").

- [ ] **Step 3: Commit**

```bash
git add notebooks/analysis.ipynb
git commit -m "feat: analysis notebook — accuracy tables, heatmap, log prob distributions, rule breakdown"
```

---

## Task 9: End-to-End Smoke Test

Verify the full pipeline works against a running Ollama instance.

- [ ] **Step 1: Confirm Ollama is running with at least one model**

```bash
curl http://localhost:11434/api/tags
```

Expected: JSON response listing available models.

- [ ] **Step 2: Run a quick smoke test**

```bash
python - <<'EOF'
from baby_reasoning.tasks.rules import RulesTask
from baby_reasoning.model import OllamaBackend
from baby_reasoning.runner import evaluate, save_results
from baby_reasoning.tasks.base import Condition

MODEL = "llama3.2:3b"  # replace with a model you have pulled

task = RulesTask()
backend = OllamaBackend(model=MODEL)
stimuli = task.canonical_stimuli()[:2]  # just 2 items for speed

results = evaluate(task, backend, Condition.ZERO_SHOT, stimuli=stimuli)
for r in results:
    print(f"rule={r.stimulus.metadata['rule']} | expected={r.stimulus.expected!r} | got={r.response.text!r} | correct={r.score.correct} | logp={r.score.logprob_correct}")

path = save_results(results, model=MODEL, task="rules", condition=Condition.ZERO_SHOT)
print(f"\nResults saved to: {path}")
EOF
```

Expected: output shows trial results and a saved path under `results/`.

- [ ] **Step 3: Final full test run**

```bash
pytest -v
```

Expected: all tests PASS.

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "feat: end-to-end evaluation framework complete — rules, hierarchical, matrix tasks with Ollama backend and notebook"
```
