#!/usr/bin/env python3
"""Preview example prompts for each task to debug formatting.

Usage:
    python preview_prompts.py                        # all tasks, n_examples=0 and 3
    python preview_prompts.py --tasks matrix_easy    # specific task(s)
    python preview_prompts.py --n_examples 0 3       # specific shot counts
    python preview_prompts.py --n 2                  # number of stimuli per task/shot combo
"""
from __future__ import annotations

import argparse
import textwrap

from baby_reasoning.cli import TASK_MAP


def _separator(char: str = "─", width: int = 72) -> str:
    return char * width


def preview(task_names: list[str], n_examples_list: list[int], n_stimuli: int) -> None:
    for task_name in task_names:
        task = TASK_MAP[task_name]()
        stimuli = task.canonical_stimuli()[:n_stimuli]

        for n_ex in n_examples_list:
            print(_separator("═"))
            print(f"  TASK: {task_name}   n_examples={n_ex}")
            print(_separator("═"))

            for i, stimulus in enumerate(stimuli):
                print(_separator())
                print(f"  Stimulus {i + 1}")
                print(_separator())
                prompt = task.build_prompt(stimulus, n_ex)
                print(prompt)
                print()
                print(f"  expected : {stimulus.expected!r}")
                if stimulus.answer_choices is not None:
                    formatted = [
                        f"{c!r} → completion {task.format_completion(stimulus, c)!r}"
                        for c in stimulus.answer_choices
                    ]
                    print(f"  choices  : {', '.join(stimulus.answer_choices)}")
                    print("  formatted completions:")
                    for line in formatted:
                        print(f"    {line}")
                meta = {k: v for k, v in stimulus.metadata.items()}
                if meta:
                    print(f"  metadata : {meta}")
                print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--tasks", nargs="+", default=list(TASK_MAP),
        choices=list(TASK_MAP), metavar="TASK",
        help=f"Tasks to preview. Choices: {', '.join(TASK_MAP)}. Default: all.",
    )
    parser.add_argument(
        "--n_examples", nargs="+", type=int, default=[0, 3], metavar="N",
        help="Shot counts to show. Default: 0 3.",
    )
    parser.add_argument(
        "--n", type=int, default=2, metavar="N",
        help="Number of stimuli to show per task/shot combo. Default: 2.",
    )
    args = parser.parse_args()
    preview(args.tasks, args.n_examples, args.n)


if __name__ == "__main__":
    main()
