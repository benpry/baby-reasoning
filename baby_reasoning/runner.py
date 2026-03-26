from __future__ import annotations

import dataclasses
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from baby_reasoning import RESULTS_DIR
from baby_reasoning.tasks.base import (
    ModelBackend,
    Stimulus,
    Task,
    TrialResult,
    TrialScore,
)


def _task_name(task: Task) -> str:
    """Derive snake-case task name from class name, e.g. RulesTask → 'rules'."""
    name = type(task).__name__
    name = name.removesuffix("Task")
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def evaluate(
    task: Task,
    backend: ModelBackend,
    n_examples: int,
    stimuli: list[Stimulus] | None = None,
) -> list[TrialResult]:
    if stimuli is None:
        stimuli = task.canonical_stimuli()

    task_name = _task_name(task)
    results = []

    for stimulus in stimuli:
        prompt = task.build_prompt(stimulus, n_examples)
        response = backend.generate(prompt)
        correct = task.score(response, stimulus)
        if stimulus.answer_choices is not None:
            logprobs = {
                c: backend.score_completion(prompt, task.format_completion(stimulus, c))
                for c in stimulus.answer_choices
            }
            logprob_correct = logprobs.get(stimulus.expected)
            valid = {c: lp for c, lp in logprobs.items() if lp is not None}
            if valid:
                max_lp = max(valid.values())
                denom = sum(math.exp(lp - max_lp) for lp in valid.values())
                lp_c = valid.get(stimulus.expected)
                prob_correct = math.exp(lp_c - max_lp) / denom if lp_c is not None else None
            else:
                prob_correct = None
            answer_logprobs = logprobs
        else:
            logprob_correct = backend.score_completion(
                prompt, task.format_completion(stimulus, stimulus.expected)
            )
            prob_correct = None
            answer_logprobs = None
        score = TrialScore(
            correct=correct,
            logprob_correct=logprob_correct,
            prob_correct=prob_correct,
            answer_logprobs=answer_logprobs,
        )
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        results.append(
            TrialResult(
                model=backend.model,
                task=task_name,
                n_examples=n_examples,
                stimulus=stimulus,
                response=response,
                score=score,
                timestamp=timestamp,
            )
        )

    return results


def save_results(
    results: list[TrialResult],
    model: str,
    task: str,
    n_examples: int,
    results_dir: Path | None = None,
    run_id: str | None = None,
) -> Path:
    if results_dir is None:
        results_dir = RESULTS_DIR

    model_tag = model.replace(":", "_").replace("/", "--")
    if run_id is None:
        run_id = (
            results[0].timestamp.replace(":", "-")
            if results
            else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%fZ")
        )

    out_dir = results_dir / model_tag / run_id / task
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{n_examples}_examples.json"

    data = [dataclasses.asdict(r) for r in results]
    out_path.write_text(json.dumps(data, indent=2))
    return out_path
