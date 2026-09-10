#!/usr/bin/env bash
# Task B ablation. Every arm shares folds, seed and preprocessing, so the only
# difference between two lines is the flag named in its tag. Run from repo root.
#
#   bash work/sweep_b.sh            # 4 arms, 5-fold each
#   FOLDS=0 bash work/sweep_b.sh    # 15% holdout, ~4x faster, for a first look
set -euo pipefail
PY=${PY:-"uv run --extra cu128 python"}
FOLDS=${FOLDS:-5}
EPOCHS=${EPOCHS:-6}
SEEDS=${SEEDS:-42}

run () {  # run <tag> <extra flags...>
  tag=$1; shift
  echo "################ $tag ################"
  $PY work/muril_b.py --tag "$tag" --folds "$FOLDS" --epochs "$EPOCHS" \
      --seeds $SEEDS "$@" 2>&1 | tee "work/${tag}.log"
}

run b_base                                     # current tuned recipe + dedupe
run b_tags   --tags                            # + gazetteer / mood / address tags
run b_focal  --loss focal --focal-gamma 2.0    # + focal loss
run b_both   --tags --loss focal --focal-gamma 2.0

echo
echo "=== macro-F1 by arm ==="
grep -H "OOF macro-F1" work/b_*.log || true
