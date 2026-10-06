# The final submission

There is no more CodaBench scoring. **One submission is made for Task A and Task B
together**, from models fine-tuned on all labelled rows: train plus the released labelled
validation. Every choice behind it is made on the fixed 15% holdouts:

| task | labelled rows | holdout | fingerprint |
|---|---|---|---|
| A | 7,193 | 1,079 | `815110ff24` |
| B | 3,532 | 530 | `f85f4f049b` |

No cross-task label derivation, and no test text in training, TAPT or screening.

The ranked notebook and ZIP checklist is in
[`FINAL_SUBMISSION_NOTEBOOKS.md`](FINAL_SUBMISSION_NOTEBOOKS.md).

All notebooks below run standalone on Kaggle: **GPU T4 x2, Internet on, nothing to
attach, no token.** Each clones the latest `task-b` and stops at 11 h, so it always saves
its outputs.

## Where things stand (2026-10-06)

| | Task B | Task A |
|---|---|---|
| model | Gemma-4-12B, 4-bit QLoRA, last-token classification head | Gemma-4-12B, same recipe |
| holdout evidence | **0.6882 / 0.7321** macro-F1/accuracy for the four-model prompt ensemble (Run 30); +0.0052 over the standard-prompt mean, but inconclusive (CI [-0.0176, +0.0273], P=.67). Four-epoch single-prompt Gemma scored **0.6872** (Run 23). | **0.8563** (Run 28, seed 42) vs char SVM 0.8117: +0.043, CI [+0.021, +0.067]; the MuRIL-era best was 0.8155; the Gemma + SVM blend is 0.007 *worse* |
| best ZIP already built | **`RECOMMENDED_b23_gemma.zip`** (Run 23: 4 epochs, 4 seeds, all 3,532 rows) | `RECOMMENDED_a28_gemma.zip` (Run 28, two healthy full fits) |

## Open questions, each answered on the holdout before the final fit

| question | notebook | owner | status | sets |
|---|---|---|---|---|
| 4 epochs or 3? | `task_b/23_gemma_epochs_seeds_final` | Aaryan | **done**: 4 epochs, 0.6872 vs 0.6792 | `EPOCHS = 4` |
| R-Drop? | `task_b/24_gemma_muril_lessons` | Aaryan | **done**: inconclusive, -0.002 [-0.027, +0.023] | `RDROP = 0` |
| LoRA rank and attention-only vs attention + MLP? | `task_b/26_gemma_lora_capacity_ablation` | Robin | ready | `--r`, `--lora-targets` |
| TAPT (LoRA next-token pretraining)? | `task_b/27_gemma_tapt_holdout` | Robin | ready; memory fixed | `TAPT` |
| definitions prompt, learning rate 2e-4? | `task_b/28_gemma_prompt_lr` | Aaryan | definitions prompt measured through Run 30; lr still pending | `PROMPT`, `LR` |
| prompt ensemble? | `task_b/30_gemma_prompt_ensemble_holdout` | Robin | **done**: 0.6882, promising but P=.67 | optional full-data ensemble |
| square-root class weights? | `task_b/31_gemma_class_weight_holdout` | Robin | **done**: 0.6822 vs balanced 0.6864; reject | `CLASS_WEIGHT = balanced` |
| full-data prompt ensemble predictions? | `task_b/32_gemma_prompt_ensemble_full` | Robin | ready | `RECOMMENDED_b32_gemma_prompt_ensemble.zip` |
| Gemma on Task A, alone or blended with the SVM? | `task_a/28_gemma_holdout_final` | Aaryan | **done**: Gemma alone, 0.8563 | Task A model |
| Task A: 4 epochs or 3? | `task_a/29_gemma_epochs` | Aaryan | ready | `EPOCHS` |
| Task A: definitions prompt? | `task_a/30_gemma_definitions_prompt` | Aaryan | ready | `PROMPT` |
| Task A: TAPT? | `task_a/31_gemma_tapt` | Aaryan | ready | `TAPT` |
| Task A: lr 2e-4? | `task_a/32_gemma_lr` | Aaryan | ready | `LR` |

These are independent and can run in parallel on separate accounts.

**Collapse guard.** Gemma occasionally collapses onto one class: Task A Run 28's seed 43
predicted Non-Hate for every row. After epoch 1 the classifier checks a sample of training
rows, and exits with code 4 if one class takes more than 95% of them. Every notebook
reruns such a job once with seed + 1000, and skips collapsed probabilities when reusing
earlier outputs.

## Today's plan (2026-10-06): two runs, then combine

The runs most likely to improve each submission use the most reliable measured gain: more
models averaged.

| task | run | settings | adds | combined final |
|---|---|---|---|---|
| A | `task_a/33_gemma_final_recipe` | `EPOCHS = 4`, `SEEDS = [44, 45]` (~3.5-4.5 h) | 2 more full-data models | with Run 28's 2: **4 models** |
| B | `task_b/32_gemma_prompt_ensemble_full` (Robin) | as is (~4-6 h) | 4 models across 2 prompts | with Run 23's 4: **8 models, 2 prompts** |

Download `submissions/<name>_test_probs.npy` from each run's output, then:

```bash
python -m hastika.common.combine_probs --task a \
    --probs a28_gemma_test_probs.npy:2 a33_gemma_test_probs.npy:2 \
    --out submissions/task_a/final/final_task_a.zip
python -m hastika.common.combine_probs --task b \
    --probs b23_gemma_test_probs.npy:4 prompt_ensemble_test_probs.npy:4 \
    --out submissions/task_b/final/final_task_b.zip
```

The combiner weights each file by its model count, checks shapes, and refuses a collapsed
file. If a run does not finish, the existing ZIPs stand: `RECOMMENDED_a28_gemma.zip` for
Task A and `RECOMMENDED_b23_gemma.zip` for Task B.

## The final run

1. **Task B.** Choose one of the two final-data recipes:

   - For the simpler four-seed single-prompt fit, open `task_b/25_gemma_final_recipe` and
     set the flags in its third code cell from the results below.
   - For the best measured Run 30 ensemble, open `task_b/32_gemma_prompt_ensemble_full`.
     It trains the two prompts with seeds 42 and 43 on all 3,532 labelled rows and writes
     `RECOMMENDED_b32_gemma_prompt_ensemble.zip`.

   The Run 25 flags are:

   | flag | default | change it when |
   |---|---|---|
   | `EPOCHS` | 4 | Run 23: 4 epochs scored 0.6872 vs 0.6792 for 3 |
   | `TAPT` | False | Run 27's TAPT - no TAPT interval is clearly above 0 |
   | `RDROP` | 0.0 | stays 0 |
   | `CLASS_WEIGHT` | `balanced` | Run 31: square-root weights lost 0.0042 macro-F1 |
   | `PROMPT` | `"short"` | definitions prompt alone was tied; Run 30's ensemble requires two prompt families |
   | `LR` | 1e-4 | Run 28 printed `lr2e-4: True` |

   Four seeds are trained on all 3,532 rows. It writes `RECOMMENDED_b25_gemma.zip`. If
   every other flag stays at its default, Run 25 reproduces Run 23's recipe, and
   `RECOMMENDED_b23_gemma.zip` remains the packaged single-prompt fallback.
2. **Task A.** Open `task_a/33_gemma_final_recipe` and set `EPOCHS`, `PROMPT`, `TAPT` and
   `LR` from Task A Runs 29-32. Each flag is printed `True` when the arm beats the base
   with P(better) >= 0.7. Four seeds are trained on all 7,193 rows. It writes
   `RECOMMENDED_a33_gemma.zip`. Its defaults are Run 28's recipe, so it is safe to run
   unchanged.
3. **Store both** under `submissions/task_a/final/` and `submissions/task_b/final/`, each
   with the ZIP, its `predictions.csv` and a README naming the notebook, the commit and
   the holdout evidence. Record them in `docs/EXPERIMENTS.md`.

Every final ZIP holds one bare `predictions.csv` with header `id,label`, one row per id of
the released test file:

| task | test file | rows |
|---|---|---|
| A | `hastika_binary_test.csv` | 806 |
| B | `hastika_multiclass_test.csv` | 396 |
