from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Stimulus:
    query: str
    expected: str
    few_shot_examples: list[tuple[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    answer_choices: list[str] | None = None


@dataclass
class ModelResponse:
    text: str
    token_logprobs: list[float] | None = None


@dataclass
class TrialScore:
    correct: bool
    logprob_correct: float | None


@dataclass
class TrialResult:
    model: str
    task: str
    n_examples: int
    stimulus: Stimulus
    response: ModelResponse
    score: TrialScore
    timestamp: str


class Task(ABC):
    @abstractmethod
    def canonical_stimuli(self) -> list[Stimulus]: ...

    @abstractmethod
    def generate_stimulus(self, n_examples: int = 3) -> Stimulus: ...

    @abstractmethod
    def score(self, response: ModelResponse, stimulus: Stimulus) -> bool: ...

    @abstractmethod
    def build_prompt(self, stimulus: Stimulus, n_examples: int) -> str: ...

    def format_completion(self, stimulus: Stimulus, choice: str) -> str:
        """Format an answer choice as it would appear appended to the prompt.

        Override in subclasses where the model expects a delimiter (e.g. space)
        between the prompt and the answer.
        """
        return choice


class ModelBackend(ABC):
    @property
    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    def generate(self, prompt: str) -> ModelResponse: ...

    @abstractmethod
    def score_completion(self, prompt: str, completion: str) -> float | None: ...
