from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock

import pytest

from baby_reasoning.cli import Config, TASK_MAP, run


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
    mocker.patch("baby_reasoning.cli.VLLMBackend")

    cfg = Config(models=["m1", "m2"], tasks=["rules"], n_examples=[0])
    run(cfg)

    assert mock_evaluate.call_count == 2  # 2 models × 1 task × 1 n_examples


def test_run_calls_save_results_for_each_combination(mocker):
    mocker.patch("baby_reasoning.cli.evaluate", return_value=[])
    mock_save = mocker.patch("baby_reasoning.cli.save_results", return_value=Path("x"))
    mocker.patch("baby_reasoning.cli.VLLMBackend")

    cfg = Config(models=["m1"], tasks=["rules", "hierarchical"], n_examples=[0, 3])
    run(cfg)

    assert mock_save.call_count == 4  # 1 model × 2 tasks × 2 n_examples


def test_run_passes_n_examples_to_evaluate(mocker):
    mock_evaluate = mocker.patch("baby_reasoning.cli.evaluate", return_value=[])
    mocker.patch("baby_reasoning.cli.save_results", return_value=Path("x"))
    mocker.patch("baby_reasoning.cli.VLLMBackend")

    cfg = Config(models=["m1"], tasks=["rules"], n_examples=[5])
    run(cfg)

    args = mock_evaluate.call_args[0]
    assert args[2] == 5  # n_examples is third positional arg


def test_run_uses_canonical_stimuli_by_default(mocker):
    mock_evaluate = mocker.patch("baby_reasoning.cli.evaluate", return_value=[])
    mocker.patch("baby_reasoning.cli.save_results", return_value=Path("x"))
    mocker.patch("baby_reasoning.cli.VLLMBackend")

    cfg = Config(models=["m1"], tasks=["rules"], n_examples=[0])
    run(cfg)

    # stimuli arg should be None so evaluate uses canonical_stimuli internally
    assert mock_evaluate.call_args[0][3] is None


def test_run_generates_n_stimuli_when_specified(mocker):
    mock_evaluate = mocker.patch("baby_reasoning.cli.evaluate", return_value=[])
    mocker.patch("baby_reasoning.cli.save_results", return_value=Path("x"))
    mocker.patch("baby_reasoning.cli.VLLMBackend")

    cfg = Config(models=["m1"], tasks=["rules"], n_examples=[0], n_stimuli=5)
    run(cfg)

    stimuli_arg = mock_evaluate.call_args[0][3]
    assert stimuli_arg is not None
    assert len(stimuli_arg) == 5


def test_run_reuses_stimuli_across_n_examples(mocker):
    """Same stimuli list is passed to evaluate for all n_examples values."""
    mock_evaluate = mocker.patch("baby_reasoning.cli.evaluate", return_value=[])
    mocker.patch("baby_reasoning.cli.save_results", return_value=Path("x"))
    mocker.patch("baby_reasoning.cli.VLLMBackend")

    cfg = Config(models=["m1"], tasks=["rules"], n_examples=[0, 3], n_stimuli=5)
    run(cfg)

    assert mock_evaluate.call_count == 2
    stimuli_0 = mock_evaluate.call_args_list[0][0][3]
    stimuli_3 = mock_evaluate.call_args_list[1][0][3]
    assert stimuli_0 is stimuli_3


def test_run_systematic_uses_systematic_stimuli(mocker):
    mock_evaluate = mocker.patch("baby_reasoning.cli.evaluate", return_value=[])
    mocker.patch("baby_reasoning.cli.save_results", return_value=Path("x"))
    mocker.patch("baby_reasoning.cli.VLLMBackend")

    cfg = Config(models=["m1"], tasks=["rules"], n_examples=[0, 3], n_stimuli=10, systematic=True)
    run(cfg)

    # systematic_stimuli for rules returns n_per_rule × 2 rules = 20 stimuli
    stimuli_arg = mock_evaluate.call_args_list[0][0][3]
    assert stimuli_arg is not None
    assert len(stimuli_arg) == 20  # 10 per rule × 2 rules


def test_run_systematic_falls_back_to_random_for_matrix(mocker):
    mock_evaluate = mocker.patch("baby_reasoning.cli.evaluate", return_value=[])
    mocker.patch("baby_reasoning.cli.save_results", return_value=Path("x"))
    mocker.patch("baby_reasoning.cli.VLLMBackend")

    cfg = Config(models=["m1"], tasks=["matrix"], n_examples=[0], n_stimuli=5, systematic=True)
    run(cfg)

    stimuli_arg = mock_evaluate.call_args[0][3]
    assert stimuli_arg is not None
    assert len(stimuli_arg) == 5


def test_run_passes_results_dir_to_save(mocker):
    mocker.patch("baby_reasoning.cli.evaluate", return_value=[])
    mock_save = mocker.patch("baby_reasoning.cli.save_results", return_value=Path("x"))
    mocker.patch("baby_reasoning.cli.VLLMBackend")

    custom_dir = Path("/tmp/custom")
    cfg = Config(models=["m1"], tasks=["rules"], n_examples=[0], results_dir=custom_dir)
    run(cfg)

    assert mock_save.call_args[1]["results_dir"] == custom_dir
