#!/bin/bash
# Submit the Pythia evaluation job array
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

mkdir -p slurm/logs
sbatch slurm/eval_model.sh
