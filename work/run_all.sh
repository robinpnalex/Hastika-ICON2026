#!/usr/bin/env bash
# HASTIKA Task A sweep. Run from the repo root: bash work/run_all.sh
# Tuned for an Ampere-or-newer GPU (bf16 autocast). MuRIL goes through the
# dedicated muril.py; the other encoders use the generic train_xlmr.py.
set -euo pipefail
PY="${PY:-python}"
FOLDS="${FOLDS:-5}"
EPOCHS="${EPOCHS:-6}"

# Floor first -- seconds, and it is a real ensemble member.
"$PY" work/baseline_svm.py 2>&1 | tee work/svm.log

# Primary model. Defaults in muril.py are already the tuned recipe; --seeds
# averages independent fine-tunes, which is the single most reliable gain at
# this dataset size.
"$PY" work/muril.py --tag muril --folds "$FOLDS" --epochs "$EPOCHS" \
    --seeds 42 1337 2>&1 | tee work/muril.log

# Diversity for the blend. Sentencepiece models keep emoji natively, so no
# --demojize. These make different errors than MuRIL, which is the point.
for spec in "twitxlmr cardiffnlp/twitter-xlm-roberta-base" \
            "xlmr xlm-roberta-base" \
            "indicbert ai4bharat/IndicBERTv2-MLM-only"; do
  set -- $spec
  echo "=== $1 ($2) ==="
  "$PY" work/train_xlmr.py --model "$2" --tag "$1" --folds "$FOLDS" \
      --epochs "$EPOCHS" --lr 3e-5 --bs 32 --max-len 128 2>&1 | tee "work/$1.log"
done

# Large models -- viable on Ampere. Lower LR, smaller batch, accumulate.
if [ "${LARGE:-1}" = "1" ]; then
  "$PY" work/muril.py --model google/muril-large-cased --tag muril-lg \
      --folds "$FOLDS" --epochs "$EPOCHS" --bs 8 --grad-accum 2 \
      --lr 1e-5 --head-lr 5e-5 --reinit-layers 3 2>&1 | tee work/muril-lg.log
fi

echo; echo "=== per-model OOF ==="; grep -H "OOF macro-F1" work/*.log || true
echo; echo "=== blend ==="; "$PY" work/ensemble.py
echo; echo "=== package ==="
"$PY" work/make_submission.py --pred work/runs/ensemble/predictions.csv --task a
