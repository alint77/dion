#!/bin/bash
set -euo pipefail
# Enqueue an optimizer benchmark sweep on the cluster via cjob.
# The benchmark runs benchmark_optimizer.py sweep, which times optimizer.step()
# in isolation (no forward/backward pass) across multiple configurations.
#
# Usage:
#   cjob_benchmark.sh [options] [-- extra_benchmark_args...]
#
# Options:
#   --config CONFIG        YAML config file (default: configs/benchmark_optimizer.yaml)
#   --name NAME            Job name (default: derived from config)
#   --priority p0|p1       Job priority (default: p0)
#   --gpus N               GPUs (default: 1)
#   --duration SECONDS     Expected runtime (default: 1800)
#   --workstream NAME      Workstream (default: phinext)
#   --image IMAGE          Container image override
#
# Examples:
#   cjob_benchmark.sh
#   cjob_benchmark.sh --config configs/dion2_160m.yaml -- --model_dim 5120
#   cjob_benchmark.sh --gpus 4 -- --fs_size 4
#   cjob_benchmark.sh -- --model_dim 5120 --n_head 32 --n_layer 20

CJOB="${CJOB:-/data/cluster-jobs/cjob}"
WORKTREE="$HOME/dion"
SHORT_USERNAME="${SHORT_USERNAME:-${USER%%@*}}"

# Defaults
CONFIG="configs/benchmark_optimizer.yaml"
NAME=""
PRIORITY="p0"
GPUS="8"
DURATION="1500"
WORKSTREAM="phinext"
CJOB_IMAGE="${CJOB_IMAGE:-}"
OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)     CONFIG="$2"; shift 2 ;;
    --name)       NAME="$2"; shift 2 ;;
    --priority)   PRIORITY="$2"; shift 2 ;;
    --gpus)       GPUS="$2"; shift 2 ;;
    --duration)   DURATION="$2"; shift 2 ;;
    --workstream) WORKSTREAM="$2"; shift 2 ;;
    --image)      CJOB_IMAGE="$2"; shift 2 ;;
    --)           shift; OVERRIDES+=("$@"); break ;;
    -*)           echo "Unknown flag: $1" >&2; exit 1 ;;
    *)            echo "Unexpected argument: $1" >&2; exit 1 ;;
  esac
done

NAME="${NAME:-${WORKSTREAM}-bench-$(basename "$CONFIG" .yaml)}"
LOCAL_RESULTS="/data/t-noahamsel/dion_results/"

CJOB_ARGS=(
    --name     "$NAME"
    --upload   "$WORKTREE"
    --priority "$PRIORITY"
    --gpus     "$GPUS"
    --duration "$DURATION"
    --workstream "$WORKSTREAM"
    --fetch-back-subdir "." "${LOCAL_RESULTS}"
    --fetch-back-exclude '*.distcp'
)

[[ -n "$CJOB_IMAGE" ]] && CJOB_ARGS+=(--image "$CJOB_IMAGE")

# Default --fs_size to match GPU count when using multiple GPUs
FS_SIZE_ARG=()
if [[ "$GPUS" -gt 1 ]]; then
    FS_SIZE_ARG=(--fs_size "$GPUS")
fi

"$CJOB" enqueue "${CJOB_ARGS[@]}" \
    -- benchmark_optimizer.py sweep \
    --config "$CONFIG" \
        "${FS_SIZE_ARG[@]}" \
        "${OVERRIDES[@]}"
