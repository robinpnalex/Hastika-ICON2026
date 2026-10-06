# Task A final submission

| file | what it is |
|---|---|
| `Project MANAS_taskA.csv` | the submitted predictions: `id,label`, 806 rows, one per id of `data/raw/hastika_binary_test.csv` |
| `RECOMMENDED_a28_gemma.zip` | the ZIP it was extracted from, byte-identical content |

**Produced by** Task A Run 28, `notebooks/task_a/28_gemma_holdout_final.ipynb`, run on
commit `ae088e6`. Log: `results/task_a/logs/run28_gemma_holdout_full.log`.

**The model:**

| | |
|---|---|
| model | Gemma-4-12B, 4-bit QLoRA, last-token classification head, 3 epochs, short prompt, lr 1e-4 |
| full-data fits averaged | 2 (seeds 42 and 43), on all 7,193 labelled rows (train + released validation, deduplicated) |
| fit health | both trained normally: final loss 0.25 and 0.27 |

**Holdout evidence** (1,079 rows, fingerprint `815110ff24`, trained on the other 6,114):

| arm | macro-F1 |
|---|---|
| **Gemma, seed 42** | **0.8563** |
| char SVM | 0.8117 |
| SVM 0.57 + MuRIL 0.43 (the MuRIL-era best) | 0.8155 |
| Gemma + SVM, 50/50 | 0.8471 |

Gemma minus SVM is +0.043, 95% CI [+0.021, +0.067]. The holdout seed-43 model collapsed;
it is not one of the two full-data models above.

**Predicted labels:** Hate 51.9%, Non-Hate 48.1% (training prior: 49.1% / 50.9%).

**Confirmed by Task A Run 34 (2026-10-06).** The same recipe, 3 epochs with two healthy
seeds, scored **0.8610** on the holdout. That is the best Task A result, ahead of the 4-epoch
variant (0.8313) and of a 3 + 4-epoch mix (0.8563).

