from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from baby_reasoning.model import VLLMBackend
from baby_reasoning.runner import evaluate, save_results
from baby_reasoning.tasks.hierarchical import HierarchicalTask
from baby_reasoning.tasks.matrix import MatrixTask
from baby_reasoning.tasks.matrix_easy import MatrixEasyTask
from baby_reasoning.tasks.rules import RulesTask

TASK_MAP = {
    "rules": RulesTask,
    "hierarchical": HierarchicalTask,
    "matrix": MatrixTask,
    "matrix_easy": MatrixEasyTask,
}

TaskName = Literal["rules", "hierarchical", "matrix", "matrix_easy"]


@dataclass
class Config:
    """Run abstract rule learning evaluations against vLLM models."""

    models: list[str]
    """One or more vLLM model names to evaluate (e.g. llama3.2 phi3)."""

    tasks: list[TaskName] = field(
        default_factory=lambda: ["rules", "hierarchical", "matrix"]
    )
    """Tasks to run. Defaults to all three."""

    n_examples: list[int] = field(default_factory=lambda: [0, 3])
    """Number of in-context examples to evaluate. Defaults to [0, 3]."""

    base_url: str = "http://localhost:8000"
    """vLLM base URL."""

    results_dir: Path | None = None
    """Override the default results directory."""

    n_stimuli: int | None = None
    """If set, generate N random stimuli instead of using the canonical set."""

    systematic: bool = False
    """Use systematic stimulus generation (balanced across rule/pattern types)."""


def _systematic_kwargs(task, n_stimuli: int, n_examples: int) -> dict:
    """Return keyword arguments for systematic_stimuli based on task type."""
    if isinstance(task, RulesTask):
        return {"n_per_rule": n_stimuli, "n_examples": n_examples}
    if isinstance(task, HierarchicalTask):
        return {"n_per_pattern": n_stimuli, "n_examples": n_examples}
    raise TypeError(f"No systematic generation for {type(task).__name__}")


def run(cfg: Config) -> None:
    from datetime import datetime, timezone

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%fZ")
    max_n_examples = max(cfg.n_examples)

    for model_name in cfg.models:
        backend = VLLMBackend(model_name, cfg.base_url)
        for task_name in cfg.tasks:
            task = TASK_MAP[task_name]()
            # Generate stimuli once per model×task, reuse across n_examples values
            if cfg.n_stimuli is not None:
                if cfg.systematic and hasattr(task, "systematic_stimuli"):
                    stimuli = task.systematic_stimuli(
                        **_systematic_kwargs(task, cfg.n_stimuli, max_n_examples)
                    )
                else:
                    stimuli = [
                        task.generate_stimulus(n_examples=max_n_examples)
                        for _ in range(cfg.n_stimuli)
                    ]
            else:
                stimuli = None
            for n_ex in cfg.n_examples:
                print(
                    f"{model_name}  {task_name}/{n_ex}_examples ... ",
                    end="",
                    flush=True,
                )
                results = evaluate(task, backend, n_ex, stimuli)
                path = save_results(
                    results,
                    model_name,
                    task_name,
                    n_ex,
                    results_dir=cfg.results_dir,
                    run_id=run_id,
                )
                n_correct = sum(r.score.correct for r in results)
                print(f"{n_correct}/{len(results)} correct → {path}")
