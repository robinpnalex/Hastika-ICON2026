#!/usr/bin/env bash
# Task A R-Drop experiment for a small NVIDIA GPU such as an RTX 3070.
#
# Runs a matched no-R-Drop control and an R-Drop arm, both with demojized MuRIL,
# then writes a valid CodaBench submission ZIP for each arm. The five-fold run
# also gives a local OOF macro-F1 comparison. Run from the repository root.
set -euo pipefail

PY="${PY:-python}"
FOLDS="${FOLDS:-5}"
EPOCHS="${EPOCHS:-6}"
SEEDS_TEXT="${SEEDS:-42}"
read -r -a SEEDS_ARRAY <<< "$SEEDS_TEXT"

mkdir -p artifacts/logs artifacts/runs

COMMON=(
  --model google/muril-base-cased
  --folds "$FOLDS"
  --epochs "$EPOCHS"
  --seeds "${SEEDS_ARRAY[@]}"
  --bs 8
  --grad-accum 2
  --eval-bs 32
  --select last
  --reinit-layers 2
)

run_arm() {
  local tag="$1"
  shift
  echo "=== $tag ==="
  "$PY" -u -m hastika.models.muril \
    --tag "$tag" "${COMMON[@]}" "$@" \
    2>&1 | tee "artifacts/logs/${tag}.log"
}

# Control: the established MuRIL regularization recipe without R-Drop.
run_arm task_a_muril_control --rdrop 0

# Variant: two stochastic dropout passes are tied with a symmetric KL penalty.
run_arm task_a_muril_rdrop --rdrop 0.5

echo
echo "=== local scores ==="
grep -H -E "OOF macro-F1|threshold-tuned OOF macro-F1" \
  artifacts/logs/task_a_muril_control.log \
  artifacts/logs/task_a_muril_rdrop.log || true

echo
echo "=== packaging submissions ==="
"$PY" -m hastika.common.submission --task a \
  --pred artifacts/runs/task_a_muril_control/predictions.csv \
  --out artifacts/runs/task_a_muril_control/submission.zip
"$PY" -m hastika.common.submission --task a \
  --pred artifacts/runs/task_a_muril_rdrop/predictions.csv \
  --out artifacts/runs/task_a_muril_rdrop/submission.zip

echo
echo "R-Drop submission: artifacts/runs/task_a_muril_rdrop/submission.zip"
echo "Control submission: artifacts/runs/task_a_muril_control/submission.zip"
echo "Upload only one after comparing local OOF scores and prediction distributions."
