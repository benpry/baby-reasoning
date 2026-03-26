#!/bin/bash
#SBATCH --job-name=baby-reasoning
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --nodelist=cocoflops2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-8
#SBATCH --output=slurm/logs/%A_%a.out
#SBATCH --error=slurm/logs/%A_%a.err

set -euo pipefail

MODELS=(
  EleutherAI/pythia-14m-deduped
  EleutherAI/pythia-70m-deduped
  EleutherAI/pythia-160m-deduped
  EleutherAI/pythia-410m-deduped
  EleutherAI/pythia-1b-deduped
  EleutherAI/pythia-1.4b-deduped
  EleutherAI/pythia-2.8b-deduped
  EleutherAI/pythia-6.9b-deduped
  EleutherAI/pythia-12b-deduped
)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

# Derive a unique port from the job ID to avoid collisions when
# multiple array tasks land on the same node.
PORT=$((8000 + SLURM_JOB_ID % 1000))

echo "=== Job $SLURM_JOB_ID | Task $SLURM_ARRAY_TASK_ID | Model: $MODEL | Port: $PORT ==="

# --- Environment setup ---
cd /sailhome/benpry/baby-reasoning
export SCR_ROOT_DIR="${SCR_ROOT_DIR:-/scr/benpry}"
export UV_PROJECT_ENVIRONMENT="${SCR_ROOT_DIR}/uv/baby-reasoning"
export HF_HOME="${SCR_ROOT_DIR}/hf_cache"
source "${UV_PROJECT_ENVIRONMENT}/bin/activate"

# --- Cleanup: kill vLLM server on exit ---
VLLM_PID=""
cleanup() {
  if [[ -n "$VLLM_PID" ]]; then
    echo "Shutting down vLLM server (PID $VLLM_PID)..."
    kill "$VLLM_PID" 2>/dev/null
    wait "$VLLM_PID" 2>/dev/null
  fi
}
trap cleanup EXIT

# --- Start vLLM server ---
echo "Starting vLLM server for $MODEL on port $PORT..."
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PORT" &
VLLM_PID=$!

# --- Wait for server readiness (up to 10 minutes) ---
MAX_WAIT=600
ELAPSED=0
echo "Waiting for vLLM server to be ready..."
until curl -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "ERROR: vLLM server process died"
    exit 1
  fi
  if [[ $ELAPSED -ge $MAX_WAIT ]]; then
    echo "ERROR: vLLM server did not become ready within ${MAX_WAIT}s"
    exit 1
  fi
  sleep 5
  ELAPSED=$((ELAPSED + 5))
done
echo "vLLM server ready after ${ELAPSED}s"

# --- Run evaluation ---
script/run \
  --models "$MODEL" \
  --tasks rules hierarchical matrix matrix_easy \
  --n-examples 0 1 3 5 7 10 20 \
  --n-stimuli 300 \
  --systematic \
  --base-url "http://localhost:${PORT}"

echo "=== Evaluation complete for $MODEL ==="
