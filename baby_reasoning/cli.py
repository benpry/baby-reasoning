from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from baby_reasoning.model import OllamaBackend
from baby_reasoning.tasks.base import Condition
from baby_reasoning.tasks.hierarchical import HierarchicalTask
from baby_reasoning.tasks.matrix import MatrixTask
from baby_reasoning.tasks.rules import RulesTask
from baby_reasoning.runner import evaluate, save_results

TASK_MAP = {
    "rules": RulesTask,
    "hierarchical": HierarchicalTask,
    "matrix": MatrixTask,
}

CONDITION_MAP = {
    "zero_shot": Condition.ZERO_SHOT,
    "few_shot": Condition.FEW_SHOT,
}

TaskName = Literal["rules", "hierarchical", "matrix"]
ConditionName = Literal["zero_shot", "few_shot"]


@dataclass
class Config:
    """Run abstract rule learning evaluations against Ollama models."""

    models: list[str]
    """One or more Ollama model names to evaluate (e.g. llama3.2 phi3)."""

    tasks: list[TaskName] = field(
        default_factory=lambda: ["rules", "hierarchical", "matrix"]
    )
    """Tasks to run. Defaults to all three."""

    conditions: list[ConditionName] = field(
        default_factory=lambda: ["zero_shot", "few_shot"]
    )
    """Experimental conditions. Defaults to both."""

    base_url: str = "http://localhost:11434"
    """Ollama base URL."""

    results_dir: Path | None = None
    """Override the default results directory."""

    n_stimuli: int | None = None
    """If set, generate N random stimuli instead of using the canonical set."""


def run(cfg: Config) -> None:
    for model_name in cfg.models:
        backend = OllamaBackend(model_name, cfg.base_url)
        for task_name in cfg.tasks:
            task = TASK_MAP[task_name]()
            for cond_name in cfg.conditions:
                condition = CONDITION_MAP[cond_name]
                stimuli = (
                    [task.generate_stimulus() for _ in range(cfg.n_stimuli)]
                    if cfg.n_stimuli is not None
                    else None
                )
                print(f"{model_name}  {task_name}/{cond_name} ... ", end="", flush=True)
                results = evaluate(task, backend, condition, stimuli)
                path = save_results(
                    results, model_name, task_name, condition, results_dir=cfg.results_dir
                )
                n_correct = sum(r.score.correct for r in results)
                print(f"{n_correct}/{len(results)} correct → {path}")
