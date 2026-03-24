# Evaluation Framework Design
_Date: 2026-03-24_

## Overview

A Python research tool for evaluating small language models (served locally via Ollama) on abstract rule learning tasks from developmental psychology. The framework is modular and reusable across many models and task types, with results stored as flat files and analyzed via a Jupyter notebook.

---

## Tasks

Three task types are in scope:

| Module | Task | Source |
|---|---|---|
| `tasks/rules.py` | Marcus rule learning (ABA, ABB, AAB patterns over syllables) | Marcus (1999) |
| `tasks/hierarchical.py` | Hierarchical equality task | Marcus (~2001); see Geiger (2022) |
| `tasks/matrix.py` | Text-based Raven's Progressive Matrices | Webb et al. (2023) |

Analogical reasoning tasks are explicitly out of scope for now.

---

## Experimental Conditions

Every task is evaluated in two conditions:

- **Zero-shot** — task instructions + query only; no examples provided
- **Few-shot** — task instructions + N labeled input→output examples + query

The number of few-shot examples is configurable per task (e.g. Marcus traditionally uses 3–6 exemplars). Condition is a parameter passed to the `evaluate()` function.

`Condition` is defined as a `str` enum:
```python
class Condition(str, Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
```

---

## Architecture

### Core Abstractions (`tasks/base.py`)

**`Stimulus` (dataclass)**
Represents one test item. `Stimulus` objects are condition-agnostic — they always carry `few_shot_examples`, which `build_prompt()` uses or ignores depending on the condition. For stimuli that have no natural few-shot examples, `few_shot_examples` defaults to `[]`; it is the caller's responsibility to ensure few-shot runs use stimuli with non-empty example lists.

```python
@dataclass
class Stimulus:
    query: str                                         # The prompt presented to the model
    expected: str                                      # The correct completion
    few_shot_examples: list[tuple[str, str]] = field(default_factory=list)  # (input, output) pairs
    metadata: dict = field(default_factory=dict)       # Rule type, source, etc.
```

**`Task` (ABC)**
Each task type subclasses this. `score()` is responsible only for the `correct: bool` determination via pattern match; `logprob_correct` is populated by the runner (see Runner section).

```python
class Task(ABC):
    @abstractmethod
    def canonical_stimuli(self) -> list[Stimulus]: ...   # Fixed curated set from data/
    @abstractmethod
    def generate_stimulus(self) -> Stimulus: ...          # Programmatic generation
    @abstractmethod
    def score(self, response: ModelResponse, stimulus: Stimulus) -> bool: ...
    @abstractmethod
    def build_prompt(self, stimulus: Stimulus, condition: Condition) -> str: ...
```

**`ModelBackend` (ABC)**
Thin adapter over Ollama (or any future backend):
```python
class ModelBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> ModelResponse: ...
    @abstractmethod
    def score_completion(self, prompt: str, completion: str) -> float | None: ...
```

`score_completion()` takes the same prompt string that was passed to `generate()` (i.e. the fully built prompt including any few-shot examples) and the correct completion as separate arguments. The implementation concatenates `prompt + completion` internally before sending to Ollama. It returns the sum of token log probs of the completion tokens, or `None` if the backend does not support log probs.

### Data Classes (`tasks/base.py`)

**`ModelResponse`**
```python
@dataclass
class ModelResponse:
    text: str
    token_logprobs: list[float] | None  # Log probs of generated tokens; None if unsupported
```

**`TrialScore`**
Assembled by the runner from `task.score()` (for `correct`) and `backend.score_completion()` (for `logprob_correct`):
```python
@dataclass
class TrialScore:
    correct: bool
    logprob_correct: float | None  # Log prob of correct completion; None if unsupported
```

**`TrialResult`**
One row of output, serialized to JSON:
```python
@dataclass
class TrialResult:
    model: str           # Ollama model tag, e.g. "llama3:8b" — use the tag consistently across runs
    task: str            # Snake-case task name, e.g. "rules", "hierarchical", "matrix"
    condition: Condition
    stimulus: Stimulus
    response: ModelResponse
    score: TrialScore
    timestamp: str       # ISO 8601 UTC with microseconds, e.g. "2026-03-24T14:32:00.123456Z"
```

Timestamps include microseconds to avoid filename collisions across rapid successive runs.

### Runner (`runner.py`)

```python
def evaluate(
    task: Task,
    backend: ModelBackend,
    condition: Condition,
    stimuli: list[Stimulus] | None = None,  # defaults to task.canonical_stimuli()
) -> list[TrialResult]: ...
```

For each stimulus, the runner:
1. Calls `task.build_prompt(stimulus, condition)` to get the full prompt string
2. Calls `backend.generate(prompt)` to get `ModelResponse`
3. Calls `task.score(response, stimulus)` to get `correct: bool`
4. Calls `backend.score_completion(prompt, stimulus.expected)` to get `logprob_correct`
5. Assembles `TrialScore(correct=correct, logprob_correct=logprob_correct)`
6. Assembles and returns `TrialResult`

Results are returned to the caller for serialization.

### Model Backend (`model.py`)

`OllamaBackend` implements `ModelBackend`:
- `generate()` calls Ollama's `/api/generate` with `logprobs: true` where supported
- `score_completion()` sends the prompt + completion to Ollama and sums the token log probs of the completion tokens; returns `None` if the model or Ollama version does not support log probs
- Log prob `None` propagates gracefully through `TrialScore` and into stored results

---

## Module Layout

```
baby_reasoning/
├── tasks/
│   ├── base.py          # ABCs, Stimulus, ModelResponse, TrialScore, TrialResult, Condition
│   ├── rules.py         # Marcus rule learning
│   ├── hierarchical.py  # Hierarchical equality
│   └── matrix.py        # Text-based matrix reasoning
├── model.py             # OllamaBackend
├── runner.py            # evaluate() function
└── __init__.py

data/
├── rules/
│   └── canonical.json
├── hierarchical/
│   └── canonical.json
└── matrix/
    └── canonical.json

results/
└── <model-tag>/<task>/<condition>/<timestamp>.json

notebooks/
└── analysis.ipynb

tests/
├── test_rules.py
├── test_hierarchical.py
├── test_matrix.py
├── test_runner.py
└── test_model.py        # Tests OllamaBackend log prob handling with a mock HTTP layer
```

---

## Stimulus Design

**Hybrid approach:** each task ships a fixed `canonical.json` for reproducibility across runs, plus a `generate_stimulus()` method for novel items.

### Canonical JSON Schema

Each `canonical.json` is a JSON array of objects. All fields map directly to `Stimulus` fields:

```json
[
  {
    "query": "ga ti ti, li na na, ta da da → wo ___",
    "expected": "le le",
    "few_shot_examples": [
      ["ga ti ti", "ga ti ti"],
      ["li na na", "li na na"]
    ],
    "metadata": {
      "rule": "ABB",
      "source": "Marcus 1999"
    }
  }
]
```

`few_shot_examples` may be `[]` for stimuli intended for zero-shot-only use. `metadata` is a free-form dict; task modules document which keys they use.

`canonical_stimuli()` loads the JSON file relative to the task module's location using `importlib.resources` or a path resolved from `__file__`, so it works regardless of the working directory.

### Rules (Marcus)
- Syllable sequences following ABA, ABB, or AAB patterns (e.g. `ga ti ti`)
- Few-shot examples are drawn from the same rule type as the query
- Scoring: lowercase + strip whitespace, check response contains expected sequence

### Hierarchical Equality
- Structured relational prompts testing whether the model recognizes hierarchical equality (e.g. same/different at nested levels)
- Scoring: match against expected relational label

### Matrix Reasoning (Webb)
- Text-encoded analogical matrices; model must complete the missing cell
- Prompt format follows Webb et al. (2023) text serialization
- Scoring: match against expected completion token(s)

---

## Scoring

The runner assembles `TrialScore` from two sources:

- `correct: bool` — returned by `task.score(response, stimulus)` via pattern match (lowercase + strip whitespace)
- `logprob_correct: float | None` — returned by `backend.score_completion(prompt, stimulus.expected)`

Raw response text is always stored in `TrialResult` regardless of score.

---

## Result Storage

One JSON file per run, path: `results/<model-tag>/<task>/<condition>/<timestamp>.json`

- `<model-tag>` is the Ollama model tag with `:` replaced by `_` for filesystem safety (e.g. `llama3_8b`)
- `<timestamp>` is ISO 8601 UTC with microseconds, with `:` replaced by `-` for filesystem compatibility (e.g. `2026-03-24T14-32-00.123456Z`)

The `timestamp` field stored inside the JSON retains standard ISO 8601 colons (e.g. `"2026-03-24T14:32:00.123456Z"`); only the filename uses hyphens. These two formats are intentionally different.

Each file is a JSON array of serialized `TrialResult` objects produced by `dataclasses.asdict()`, which yields nested JSON objects for `stimulus`, `response`, and `score` fields. No database; files are human-readable and easy to load in the notebook.

The notebook flattens the nested structure on load, promoting these fields to top-level DataFrame columns: `model`, `task`, `condition`, `query`, `expected`, `few_shot_examples`, `metadata`, `response_text`, `token_logprobs`, `correct`, `logprob_correct`, `timestamp`. `few_shot_examples` and `metadata` are retained as-is (list and dict columns respectively) and not further exploded.

---

## Reporting

`notebooks/analysis.ipynb` loads all result files from `results/` into a flat pandas DataFrame. Expected columns after deserialization (see Result Storage section for the full flattened schema):

`model`, `task`, `condition`, `query`, `expected`, `few_shot_examples`, `metadata`, `response_text`, `token_logprobs`, `correct`, `logprob_correct`, `timestamp`

The notebook produces:
- Accuracy tables (model × task × condition)
- Log probability distributions per condition
- Per-task breakdown by rule/pattern type (from `stimulus.metadata`)

---

## Testing

Test files and coverage:

- `test_rules.py` — canonical stimuli load, `generate_stimulus()` validity, `score()` exact match and near-miss, `build_prompt()` for both conditions
- `test_hierarchical.py` — same structure as above
- `test_matrix.py` — same structure as above
- `test_runner.py` — integration test using a stub `ModelBackend`; verifies `TrialScore` assembly, `logprob_correct` propagation, `None` handling
- `test_model.py` — `OllamaBackend` unit tests using a mock HTTP layer; covers log prob summing, `None` return when unsupported
