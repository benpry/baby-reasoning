from __future__ import annotations

import dataclasses
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from baby_reasoning import RESULTS_DIR
from baby_reasoning.tasks.base import (
    Condition,
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
        if stimulus.answer_choices is not None:
            logprobs = {c: backend.score_completion(prompt, c) for c in stimulus.answer_choices}
            best = max(logprobs, key=lambda c: logprobs[c] if logprobs[c] is not None else float("-inf"))
            correct = best == stimulus.expected
            logprob_correct = logprobs.get(stimulus.expected)
        else:
            correct = task.score(response, stimulus)
            logprob_correct = backend.score_completion(prompt, stimulus.expected)
        score = TrialScore(correct=correct, logprob_correct=logprob_correct)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        results.append(
            TrialResult(
                model=backend.model,
                task=task_name,
                condition=condition,
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
    condition: Condition,
    results_dir: Path | None = None,
) -> Path:
    if results_dir is None:
        results_dir = RESULTS_DIR

    model_tag = model.replace(":", "_")
    ts = (
        results[0].timestamp.replace(":", "-")
        if results
        else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%fZ")
    )

    out_dir = results_dir / model_tag / task / condition.value
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ts}.json"

    data = [dataclasses.asdict(r) for r in results]
    out_path.write_text(json.dumps(data, indent=2))
    return out_path
