from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock

import pytest

from baby_reasoning.cli import Config, TASK_MAP, run
from baby_reasoning.tasks.base import Condition


# ---------------------------------------------------------------------------
# TASK_MAP
# ---------------------------------------------------------------------------

def test_task_map_has_all_tasks():
    assert set(TASK_MAP) == {"rules", "hierarchical", "matrix"}


def test_task_map_instantiates():
    for name, cls in TASK_MAP.items():
        task = cls()
        assert hasattr(task, "canonical_stimuli")


# ---------------------------------------------------------------------------
# run() — combinations
# ---------------------------------------------------------------------------

def test_run_calls_evaluate_for_each_combination(mocker):
    mock_evaluate = mocker.patch("baby_reasoning.cli.evaluate", return_value=[])
    mocker.patch("baby_reasoning.cli.save_results", return_value=Path("x"))
    mocker.patch("baby_reasoning.cli.OllamaBackend")

    cfg = Config(models=["m1", "m2"], tasks=["rules"], conditions=["zero_shot"])
    run(cfg)

    assert mock_evaluate.call_count == 2  # 2 models × 1 task × 1 condition


def test_run_calls_save_results_for_each_combination(mocker):
    mocker.patch("baby_reasoning.cli.evaluate", return_value=[])
    mock_save = mocker.patch("baby_reasoning.cli.save_results", return_value=Path("x"))
    mocker.patch("baby_reasoning.cli.OllamaBackend")

    cfg = Config(models=["m1"], tasks=["rules", "hierarchical"], conditions=["zero_shot", "few_shot"])
    run(cfg)

    assert mock_save.call_count == 4  # 1 model × 2 tasks × 2 conditions


def test_run_passes_condition_enum_to_evaluate(mocker):
    mock_evaluate = mocker.patch("baby_reasoning.cli.evaluate", return_value=[])
    mocker.patch("baby_reasoning.cli.save_results", return_value=Path("x"))
    mocker.patch("baby_reasoning.cli.OllamaBackend")

    cfg = Config(models=["m1"], tasks=["rules"], conditions=["few_shot"])
    run(cfg)

    _, kwargs = mock_evaluate.call_args
    args = mock_evaluate.call_args[0]
    assert Condition.FEW_SHOT in args


def test_run_uses_canonical_stimuli_by_default(mocker):
    mock_evaluate = mocker.patch("baby_reasoning.cli.evaluate", return_value=[])
    mocker.patch("baby_reasoning.cli.save_results", return_value=Path("x"))
    mocker.patch("baby_reasoning.cli.OllamaBackend")

    cfg = Config(models=["m1"], tasks=["rules"], conditions=["zero_shot"])
    run(cfg)

    # stimuli arg should be None so evaluate uses canonical_stimuli internally
    assert mock_evaluate.call_args[0][3] is None


def test_run_generates_n_stimuli_when_specified(mocker):
    mock_evaluate = mocker.patch("baby_reasoning.cli.evaluate", return_value=[])
    mocker.patch("baby_reasoning.cli.save_results", return_value=Path("x"))
    mocker.patch("baby_reasoning.cli.OllamaBackend")

    cfg = Config(models=["m1"], tasks=["rules"], conditions=["zero_shot"], n_stimuli=5)
    run(cfg)

    stimuli_arg = mock_evaluate.call_args[0][3]
    assert stimuli_arg is not None
    assert len(stimuli_arg) == 5


def test_run_passes_results_dir_to_save(mocker):
    mocker.patch("baby_reasoning.cli.evaluate", return_value=[])
    mock_save = mocker.patch("baby_reasoning.cli.save_results", return_value=Path("x"))
    mocker.patch("baby_reasoning.cli.OllamaBackend")

    custom_dir = Path("/tmp/custom")
    cfg = Config(models=["m1"], tasks=["rules"], conditions=["zero_shot"], results_dir=custom_dir)
    run(cfg)

    assert mock_save.call_args[1]["results_dir"] == custom_dir
