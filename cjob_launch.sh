#!/bin/bash
set -euo pipefail
# Enqueue a dion training job on the cluster via cjob.
#
# Usage:
#   cjob_launch.sh [options] [-- extra_train_args...]
#
# Options:
#   --config CONFIG        YAML config file (default: configs/muon_160m.yaml)
#   --name NAME            Job name (default: derived from config)
#   --priority p0|p1       Job priority (default: p0)
#   --gpus N               GPUs (default: 4)
#   --duration SECONDS     Expected runtime (default: 4200)
#   --workstream NAME      Workstream (default: phinext)
#   --image IMAGE          Container image override
#
# Examples:
#   cjob_launch.sh --config configs/dion2_160m.yaml
#   cjob_launch.sh --config configs/muon_160m.yaml -- --no_triton
#   cjob_launch.sh --config configs/dion2_160m.yaml --gpus 8 -- --split_heads

CJOB="${CJOB:-/data/cluster-jobs/cjob}"
WORKTREE="$HOME/dion"
SHORT_USERNAME="${SHORT_USERNAME:-${USER%%@*}}"

# Defaults
CONFIG="configs/muon_160m.yaml"
NAME=""
PRIORITY="p0"
GPUS="4"
DURATION="4200"
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

NAME="${NAME:-${WORKSTREAM}-dion-$(basename "$CONFIG" .yaml)}"
LOCAL_RESULTS="/data/${SHORT_USERNAME}/dion_results"

CJOB_ARGS=(
    --name     "$NAME"
    --upload   "$WORKTREE"
    --priority "$PRIORITY"
    --gpus     "$GPUS"
    --duration "$DURATION"
    --workstream "$WORKSTREAM"
    --env      "WANDB_API_KEY=${WANDB_API_KEY:-}"
    --env      "WANDB_BASE_URL=${WANDB_BASE_URL:-https://microsoft-research.wandb.io}"
    --env      "WANDB_PROJECT=dion-repo"
    --fetch-back-subdir "logs" "${LOCAL_RESULTS}/"
    --fetch-back-exclude '*.distcp'
)

[[ -n "$CJOB_IMAGE" ]] && CJOB_ARGS+=(--image "$CJOB_IMAGE")

TRAIN_ARGS=(
    --config "$CONFIG"
    --data_dir /data/datafromoldb200/msraif-shared-pvc-local-msraif-shared-01/kwangjunahn/fineweb100B/
    --wandb_project_name "dion-repo"
)

"$CJOB" enqueue "${CJOB_ARGS[@]}" \
    -- train.py \
        "${TRAIN_ARGS[@]}" \
        "${OVERRIDES[@]}"
