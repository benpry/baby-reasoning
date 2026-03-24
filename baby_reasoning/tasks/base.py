from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Condition(str, Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"


@dataclass
class Stimulus:
    query: str
    expected: str
    few_shot_examples: list[tuple[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


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
    @property
    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    def generate(self, prompt: str) -> ModelResponse: ...

    @abstractmethod
    def score_completion(self, prompt: str, completion: str) -> float | None: ...
