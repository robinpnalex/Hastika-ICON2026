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

## Latest (2026-10-14): TAPT works

**Task A best-supported ZIP:** `RECOMMENDED_a33_gemma.zip` from the **TAPT** Run 33 (commit
`de2fc07`): four full-data TAPT models. Its holdout twin measured **0.8702**, against
0.8582-0.8610 for every no-TAPT ensemble.

**The TAPT line, eight notebooks.** Run each holdout twin beside its full-data fit:

| | holdout twin | full data | ZIP |
|---|---|---|---|
| A, TAPT | `task_a/35_tapt_final_holdout_twin` | `task_a/33_gemma_final_recipe` | `RECOMMENDED_a33_gemma_tapt.zip` |
| A, TAPT + R-Drop | `task_a/36_tapt_rdrop_holdout_twin` | `task_a/37_tapt_rdrop_final` | `RECOMMENDED_a37_gemma_tapt_rdrop.zip` |
| B, TAPT (2 epochs) | `task_b/35_tapt_final_holdout_twin` | `task_b/25_gemma_final_recipe` | `RECOMMENDED_b25_gemma_tapt.zip` |
| B, TAPT + R-Drop | `task_b/36_tapt_rdrop_holdout_twin` | `task_b/37_tapt_rdrop_final` | `RECOMMENDED_b37_gemma_tapt_rdrop.zip` |

Every twin also writes `<run>_holdout_models_test_probs.npy`. Average those with the
full-data probabilities using `python -m hastika.common.combine_probs` to get more models.
All eight are TAPT-only: if TAPT fails, they stop and write nothing.

## FINAL RECIPES (2026-10-16): all runs complete

| | Task A | Task B |
|---|---|---|
| base | Gemma-4-12B, 4-bit QLoRA, last-token classification head | same |
| TAPT | 1 epoch, LoRA next-token, on labelled comment text + `offenseval_kn` | 2 epochs |
| classifier | 3 epochs, lr 1e-4, balanced class weights, short prompt | 4 epochs |
| ensemble | **12 models** (Run 33 x4, Run 37 x2 R-Drop, Run 38 t1-t3 x2), equal weight | **8 models** (Run 38 t1-t3 x2, Run 37 x2 R-Drop), equal weight |
| holdout evidence | 0.8721 (vs 0.8610 without TAPT) | 0.7026-0.7056 (vs 0.6872 without TAPT) |

The `combine_probs` commands are at the end of `EXPERIMENTS.md`. Every member has run;
only the combination step is left.

**One experiment still to run: Task A Run 40** (Qwen3-8B as a second model family, holdout
only). It is self-contained: it trains its own Gemma TAPT x2 reference, about 6 h. Only if its verdict is SIGNIFICANT is a full-data Qwen fit
added to the Task A ensemble, with a Task B version built the same way. Otherwise the
recipes above are final. The review of what else could be tried, and why each is closed,
is at the end of `EXPERIMENTS.md`.

## Remaining plan: as few runs as possible (2026-10-15)

**Principle.** Generalise, do not tune to the holdout. Every recipe is already fixed:

| setting | Task A | Task B |
|---|---|---|
| TAPT epochs | 1 | 2 |
| classifier epochs | 3 | 4 |
| prompt, lr, class weights | short, 1e-4, balanced | short, 1e-4, balanced |

From here on, the only thing that grows is the number of independently trained TAPT
models averaged together. Averaging reduces variance without fitting anything to the
holdout, so it is the safe way to improve generalisation. There is no seed picking, and no
holdout-tuned weights or thresholds: every model gets equal weight. Holdout runs are kept to
the minimum needed to confirm each recipe once.

**Status:**

| task | done | running | still to run |
|---|---|---|---|
| A | Runs 33, 35, 36, 37, 38 t1-t3, 39 t1 | -- | nothing: all done |
| B | Runs 36, 37, 38 t1-t3, 39 t1 | -- | nothing: all done |

**Why no more holdout bag sessions.** Task A's bag holdout (Run 39 t1, 0.8675) already
confirms that an independent TAPT run reproduces the gain. Task B gets exactly one, Run 39
t1. Further holdout sessions would only re-measure the averaging curve, so the remaining
bag sessions are full data only. They add models, not measurements.

**Final ensembles, equal weight per model:**

| task | members |
|---|---|
| A | Run 33 (4) + **Run 37 (2, TAPT + R-Drop; Run 36: the mix scored 0.8721)** + Run 38 t1, t2, t3 (2 each) = **12 models from 5 TAPT runs** |
| B | Run 38 t1, t2, t3 (2 each) + **Run 37 (2, TAPT + R-Drop; Run 36: 0.7026)** = **8 models from 4 TAPT runs** |

Combine them with `python -m hastika.common.combine_probs`, weighting each file by its
model count.

## The one remaining idea: a second model family (Task A Run 40)

Every other untried change is expected to fall inside the holdout noise, so it is left out.
Run 40 trains Qwen3-8B-Base with Gemma's recipe and tests whether adding it to the Gemma
ensemble, at equal weight per model, beats Gemma alone. A full-data Qwen fit is built only if
the gain is significant (CI lower bound above 0). Attach Run 36's output when running it.

## Task A as of 2026-10-15

| holdout ensemble | macro-F1 |
|---|---|
| TAPT x4, one TAPT run | **0.8702** |
| TAPT x2, a second TAPT run (bag t1) | 0.8675 |
| best no-TAPT | 0.8610 |

**Best submission to build now:** Run 33's 4 TAPT models plus Run 38 t1's 2:

```bash
python -m hastika.common.combine_probs --task a \
    --probs a33_gemma_test_probs.npy:4 a38_bag_t1_test_probs.npy:2 \
    --out submissions/task_a/final/final_task_a_tapt6.zip
```

Still to run for Task A:
- **Run 36,** which checks whether R-Drop adds to TAPT. Until it does, Run 37's ZIP stays out.
- **Bag sessions t2-t4** (Runs 38 and 39), to grow the ensemble.

## Next: the TAPT bag (most reliable gain)

Averaging is where TAPT gains (1 model about 0.852, 4 models 0.8702), so the next step is
more TAPT models built from *different* TAPT runs. Run each pair on its own account, one
session per `TAPT_SEED` (1, 2, 3, 4 ...):

| task | full data | holdout counterpart |
|---|---|---|
| A | `task_a/38_tapt_bag_full` | `task_a/39_tapt_bag_holdout` |
| B | `task_b/38_tapt_bag_full` | `task_b/39_tapt_bag_holdout` |

Score the holdout bag with `python -m hastika.common.ensemble_curve`, and combine the
full-data sessions with `python -m hastika.common.combine_probs`.

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
| Task A: 4 epochs or 3? | `task_a/34_final_holdout_check` | Aaryan | **done**: 3 epochs (0.8610) beats 4 (0.8313) | `EPOCHS = 3` |
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

**Did it work? Each final has a holdout twin.** It trains the same members on the 85%
split and scores the combination against the current best on the fixed holdout. Run each
twin in parallel with its full-data fit:

| final | twin | compares | attach to save time |
|---|---|---|---|
| Task A: Run 28 + Run 33 | `task_a/34_final_holdout_check` | four-model combination vs Run 28's recipe | Run 28's output |
| Task B: Run 23 + Run 32 | `task_b/33_final_holdout_check` | eight-model combination vs Run 23's recipe | Runs 23 and 30 |

Each prints a verdict. If it says *probably WORSE*, submit the current best ZIP instead.

The combiner weights each file by its model count, checks shapes, and refuses a collapsed
file. If a run does not finish, the existing ZIPs stand: `RECOMMENDED_a28_gemma.zip` for
Task A and `RECOMMENDED_b23_gemma.zip` for Task B.

## The TAPT option (2026-10-06)

TAPT was the largest MuRIL gain on both tasks. It is applied on top of each task's current
best recipe, with a holdout check and a submission run in parallel:

| task | holdout check | submission (full data) |
|---|---|---|
| A | **`task_a/35_tapt_final_holdout_twin`**, the exact twin of Run 33: TAPT, then four 3-epoch classifiers, vs Run 28's recipe | `task_a/33_gemma_final_recipe`, as is (`TAPT = True` is the default) |
| B | **`task_b/35_tapt_final_holdout_twin`**, the exact twin of Run 25: TAPT, then four 4-epoch classifiers, vs Run 23's recipe | `task_b/25_gemma_final_recipe`, as is (`TAPT = True` and `EPOCHS = 4` are the defaults) |

These are TAPT-only. If TAPT cannot run, the notebook stops with an error and writes no ZIP; it never substitutes a non-TAPT model. Each twin's verdict compares TAPT x2 with the current best recipe x2, a fair test. Submit the TAPT ZIP only if the verdict says it BEATS or is probably better. Its TAPT x4 line is the level to expect from the submission. Otherwise submit the
current best ZIPs, `RECOMMENDED_a28_gemma.zip` and `RECOMMENDED_b23_gemma.zip`.

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
