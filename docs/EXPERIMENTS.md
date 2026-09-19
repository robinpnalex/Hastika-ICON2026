# Training Results Log

This is the experiment ledger for the project. Ideas not yet run, and ideas measured and
rejected, live in [`IDEAS.md`](IDEAS.md). It separates completed measurements,
provisional findings, and planned runs. Local Task B scores are unbiased OOF macro-F1
unless explicitly marked as a holdout or full fit; full-data fits have no local score.

* **Task A** — binary `Hate` / `Non-Hate` classification.
* **Task B** — six-way target classification: `Gender`, `Geo-political`, `Others`,
  `Political`, `Religion`, and `Violence`.

## Current status — 2026-09-18

| item | current state |
|---|---|
| Best Task A result | Runs 9 and 11 tied at **`0.8187`** / **`0.8188`** macro-F1 on validation A |
| Best Task A component | TAPT MuRIL alone, `0.8128` five-fold OOF, against the `0.8073` TF-IDF floor |
| Best Task B local result | One-layer reinitialization, `0.614` averaged five-fold OOF macro-F1, Run 6 |
| Best recorded Task B CodaBench result | `b_reinit1_rdrop_full`, **`0.6410`** — confirmed Run 9 |
| Current Task B candidate | `b_reinit1_rdrop_full`, R-Drop plus one-layer reinitialization |
| Next Task B step | Optional: Run 10, the context-conditional decode correction, ~2.6 h |
| Current Task A work | Run 10 completed and rejected at 0.71; Runs 5, 7 and 11--15 written and unrun |
| Next Task A step | Combine the winners: TAPT + 10 epochs + two reinitialized layers, five-fold |
| Current branch | `task-b` |
| Official validation size | 395 rows with hidden labels; score via CodaBench |

Run 9 is now the strongest confirmed Task B submission: the one-layer full-data R-Drop fit
scored `0.6410` on CodaBench, improving on Run 8's `0.6299`. Runs 6 and 7 support the
one-layer choice, while Run 9 supports adding R-Drop to the full-data recipe.

## Experiment roadmap

| run | date/status | notebook or runner | question | result/status |
|---|---|---|---|---|
| Task A 1--4 | completed | Task A training scripts | baseline, demojization, and TF-IDF comparisons | best: `0.8103` |
| Task A 5 | 2026-09-19, ready; not run | `05_rdrop_full_data.ipynb` | test R-Drop against a matched control, both on all 6,401 rows | four CodaBench ZIPs; no local score by construction |
| Task A 6 | 2026-09-17, completed | MuRIL validation submission | compare MuRIL with the TF-IDF result on validation A | **0.8163 macro-F1, 0.8164 accuracy** |
| Task A 7 | 2026-09-17, in progress | `07_tapt_holdout.ipynb` | test Task-A-domain TAPT before demojized MuRIL fine-tuning | awaiting holdout results |
| Task A 8 | 2026-09-17, completed | `08_muril_tfidf_ensemble.ipynb` | test a MuRIL + TF-IDF OOF blend | **0.7890 CodaBench macro-F1, 0.7891 accuracy**; not retained |
| Task A 9 | 2026-09-18, completed | `08_muril_tfidf_ensemble.ipynb` | train both ensemble components on all data | **0.8187 CodaBench macro-F1, 0.8189 accuracy** |
| Task A 10 | 2026-09-19, completed | `10_frozen_embeddings_svm.ipynb` | test an RBF SVM on frozen MuRIL embeddings | **0.71 macro-F1, 0.70 accuracy**; rejected, and rejected as a blend member too |
| Task A 11 | 2026-09-19, completed | `11_reinit1_full_data.ipynb` | one reinitialized layer, with the blend weight and threshold refitted to match | **0.8188 macro-F1, 0.8189 accuracy** — a tie with Run 9's 0.8187 |
| Task A 16 | 2026-09-20, stage 1 completed | `16_funnel.ipynb` | which component-level ideas actually help? Six arms screened | **epochs10 +0.028, large +0.021, reinit2 +0.016; both external arms dead** |
| Task A 12 | 2026-09-20, completed | `12_tapt_oof.ipynb` | does TAPT help Task A, as it did Task B at +2.9? | **yes: 0.8128 vs 0.7894, +0.0234** |
| Task A 13 | 2026-09-19, ready; not run | `13_external_labels.ipynb` | can the external corpus's labels be trained on? control vs mix vs stage | 5-fold OOF; rules question attached |
| Task A 14 | 2026-09-19, ready; not run | `14_third_member_blend.ipynb` | does XLM-R as a third ensemble member help? | 5-fold OOF, nested three-way blend |
| Task A 15 | 2026-09-19, ready; not run | `15_capacity_and_schedule.ipynb` | is the recipe underfitting? MuRIL-large and 10 epochs | 5-fold OOF; collapse check first |
| Task B 1 | 2026-09-12, completed | `01_baseline_sweep.ipynb` | which encoder/loss is useful? | TAPT MuRIL `0.6013` OOF; submitted `0.5922` |
| Task B 2 | 2026-09-13, completed | `02_fullfit_sweep.ipynb` | full-data five-seed versions | `f_tapt` scored `0.6007` on CodaBench, inferred |
| Task B 3 | 2026-09-13--14, completed | `03_factorial_grid.ipynb` | more TAPT text, vocabulary extension, auxiliary head | `D0_V0_noaux` remained best |
| Task B 4 | not run | `04_full_data_fit.ipynb` | original two-layer full-data fit | superseded by the one-layer candidate |
| Task B 5 | 2026-09-16, completed | `05_reinit_one_layer.ipynb` | one versus two reinitialized layers | one layer: `0.6102` vs `0.6013` |
| Task B 6 | 2026-09-17, completed | `06_reinit_confirmation.ipynb` | confirm one layer at seeds 43/44 | one layer wins averaged OOF: `0.614` vs `0.612` |
| Task B 7 | 2026-09-17, completed | `07_no_reinit_ablation.ipynb` | compare one layer with no reinitialization | one layer wins averaged OOF: `0.614` vs `0.584` |
| Task B 8 | 2026-09-17, completed | `08_full_data_fit_reinit1.ipynb` | train current candidate on all data and infer validation | **0.6299 CodaBench** |
| Task B 9 | 2026-09-17, completed | `09_rdrop_one_layer.ipynb` | submit full-data R-Drop on the current one-layer recipe | **0.6410 CodaBench macro-F1, 0.6937 accuracy** |
| Task B 10 | ready; not run | `10_context_decode.ipynb` | does a violent word mean Violence only when no target group is named? | 5-fold OOF, nested; ZIP written only if it wins |
| Task B 11 | ready; not run | `11_context_tags.ipynb` | compare the winning recipe with fold-safe topic, mood and address tags | matched 5-fold OOF, ~3.5--4 h |
| Task B 12 | ready; not run | `12_full_data_context_tags.ipynb` | train the context-tagged winning recipe on all labelled rows and package a submission | CodaBench pending, ~2--3 h |
| Task B 13 | ready; not run | `experiments/task_b/stack_full.py` | four full-data arms stacked on the 0.6410 recipe: 10 epochs, unchanged control, tags, no-FGM | CodaBench pending, ~8.9 h |

## Ordered next steps

1. Run [Task A Run 10](../notebooks/task_a/10_frozen_embeddings_svm.ipynb) and compare
   its fixed-holdout score with the existing MuRIL recipe.
2. Finish [Task A Run 7](../notebooks/task_a/07_tapt_holdout.ipynb) and compare
   the stock and TAPT MuRIL holdout scores.
3. Treat the Run 8 ensemble as rejected for submission: its local OOF macro-F1 was
   `0.8233`, but its official CodaBench score was only `0.7890`.
4. If TAPT improves the matched local result, train a full-data, multi-seed candidate
   and submit it against the current Task A best of `0.8187`.
5. Keep `b_reinit1_rdrop_full` as the Task B candidate and focus new GPU budget on
   Task A rather than further reinitialization ablations.
7. If Task B gets another session, run [Task B Run 10](../notebooks/task_b/10_context_decode.ipynb)
   or [Task B Run 11](../notebooks/task_b/11_context_tags.ipynb). Run 10 tests a cheap
   post-training decoder; Run 11 tests whether fold-safe context tags improve the model
   itself. Run 11 is the stronger next model experiment, while Run 10 leaves behind
   reusable `oof_probs.npy` for future decoder ideas.

Do not compare a full-data fit to a local OOF score as though they were the same
measurement. A full-data fit has no local F1; its only evaluation is CodaBench. Also treat
small differences on the 395-row official validation set cautiously.

Task A and Task B macro-F1 are **not comparable**: Task A is two classes and Task B is six, so 0.80 on one is not better than 0.60 on the other.

## Task A

## Experiment 1: Baseline MuRIL
* **Model:** `google/muril-base-cased`
* **Preprocessing:** Default (HTML cleaning, mojibake repair)
* **Epochs:** 6
* **Best Epoch:** 6
* **Best Macro F1:** 0.7937
* **Output Directory:** `artifacts/checkpoints/muril_task_a`

## Experiment 2: Demojized MuRIL
* **Model:** `google/muril-base-cased`
* **Preprocessing:** Default + Demojized (Emojis converted to English text descriptions)
* **Epochs:** 6
* **Best Epoch:** 5
* **Best Macro F1:** 0.8023
* **Output Directory:** `artifacts/checkpoints/muril_task_a_demojized`
* **Notes:** Converting emojis to their text equivalents yielded an improvement of ~0.9% in the macro F1 score.

## Experiment 3: TF-IDF + LinearSVC Baseline
* **Model:** `TfidfVectorizer` (Word + Char n-grams) + `LinearSVC`
* **Preprocessing:** Default (HTML cleaning, mojibake repair)
* **5-Fold CV Macro F1:** 0.8039
* **Output Directory:** `artifacts/runs/svm/`
* **Notes:** Surprisingly outperformed Baseline MuRIL, establishing a strong floor.

## Experiment 4: Demojized TF-IDF + LinearSVC Baseline
* **Model:** `TfidfVectorizer` (Word + Char n-grams) + `LinearSVC`
* **Preprocessing:** Default + Demojized (Emojis converted to English text descriptions)
* **5-Fold CV Macro F1:** 0.8103
* **Output Directory:** `artifacts/runs/svm_demojize/`
* **Notes:** Previously best performing model. It has now been surpassed by the MuRIL validation submission recorded below.

## Experiment 5: full-data R-Drop against a matched control, notebook created 2026-09-19

[The R-Drop notebook](../notebooks/task_a/05_rdrop_full_data.ipynb) trains two arms on all
6,401 deduplicated rows with `--folds 1`, five seeds each, differing by one flag only:

| arm | `--rdrop` | everything else |
|---|---|---|
| control | `0` | demojized MuRIL, two-layer reinitialization, 6 epochs, effective batch 16, `--select last`, seeds 42--46 |
| variant | `0.5` | identical |

A matched control is trained rather than reusing Run 9's `0.8187`, because that
submission's MuRIL component used a **single seed**. Comparing a five-seed R-Drop arm
against it would change two things at once, and the 806-row validation set cannot
separate them.

The notebook writes four ZIPs: each MuRIL arm alone, and each blended with a full-fit
TF-IDF/SVM at Run 8's fixed 57/43 weights, so the result is also comparable with the
current best. Those weights are carried over unchanged and deliberately not re-optimized,
because there is no OOF pass here to fit them on honestly.

Expected runtime is about 6.5 hours: roughly 33 minutes per control seed and 45 per
R-Drop seed, since R-Drop adds a second forward pass. Upload `task_a_control_blend.zip`
first; its score against `0.8187` shows how far the 806-row set moves on its own, and
only then is the control-versus-R-Drop gap readable.

Neither arm has a local macro-F1 and neither can have one, because every labelled row is
in training. `experiments/task_a/run_rdrop.sh` remains the five-fold version of the same
comparison, which does produce an OOF score at roughly five times the cost.

## Experiment 6: MuRIL validation submission, 2026-09-17
* **Model:** MuRIL
* **Evaluation:** Task A validation A on CodaBench
* **Macro F1:** **0.8163**
* **Accuracy:** **0.8164**
* **Notes:** Best Task A result recorded so far, improving on demojized TF-IDF + LinearSVC (`0.8103`) by `0.0060` macro-F1. The exact flags for the submitted artifact were not provided, so this entry does not assume whether it used R-Drop or another variant.

## Task A follow-up

Run 5 above is the full-data pair that gets submitted. `experiments/task_a/run_rdrop.sh`
is the five-fold version of the same control-versus-R-Drop comparison: slower, but it
produces an OOF macro-F1, so it can answer whether R-Drop helps without spending a
CodaBench slot. Neither replaces the confirmed `0.8187` ensemble as the candidate until a
score says otherwise.

## Experiment 7: Task-A-domain TAPT + demojized MuRIL, notebook created 2026-09-17

[The Kaggle notebook](../notebooks/task_a/07_tapt_holdout.ipynb) compares two
matched Task A arms:

| arm | encoder initialization | classifier fine-tuning |
|---|---|---|
| control | stock `google/muril-base-cased` | demojized MuRIL, two-layer reinitialization |
| TAPT | MuRIL adapted with masked-LM training | the same demojized MuRIL recipe |

The notebook recreates the fixed deduplicate-first, stratified 85/15 split with seed 42.
TAPT sees only the 85% classifier-training-side comments plus permitted OffensEval Kannada
text; the Task A holdout, Task A validation inputs, labels, and Task B files are excluded.
Both arms use seed 42, six classifier epochs, effective batch size 16, FGM, EMA,
`--select last`, and produce validated Task A submission ZIPs. The TAPT stage runs for
eight MLM epochs with a 5% held-out perplexity check. Expected runtime is approximately
2--4 hours on a T4 or RTX 3070.

Outputs are `task_a_muril_control.zip`, `task_a_muril_tapt.zip`, logs, holdout
probabilities, and `task_a_tapt_summary.csv` in the Kaggle Output tab. The experiment is
created and code-validated, but has not been run yet. Its purpose is to decide whether
TAPT is worth a later full-data submission against the current `0.8163` MuRIL result; it
does not assume that the Task B TAPT gain transfers to Task A.

## Experiment 8: demojized MuRIL + TF-IDF OOF ensemble, 2026-09-17

[The ensemble notebook](../notebooks/task_a/08_muril_tfidf_ensemble.ipynb) trains
demojized TF-IDF + calibrated LinearSVC and demojized MuRIL on the same deduplicate-first,
five-fold split with split seed 42. It fits blend weights from OOF probabilities and uses
a nested weight-search estimate to check whether any apparent blend gain survives
out-of-sample evaluation. It also packages the two individual models and the ensemble
as validated Task A ZIP candidates. The completed run used a Tesla T4 and finished all
five MuRIL folds, the blend search, and ZIP validation.

The individual OOF scores were `0.8073` for TF-IDF/SVM and `0.7932` for MuRIL. The
selected blend was 57% SVM and 43% MuRIL, scoring `0.8233` OOF macro-F1 (`0.8230`
nested estimate). However, the submitted ensemble scored only **`0.7890` macro-F1** and
`0.7891` accuracy on CodaBench, below the confirmed MuRIL result of `0.8163`/`0.8164`.
This local-to-official mismatch means the blend is not a viable candidate; retain the
MuRIL submission as the Task A baseline and do not spend more GPU time rerunning this
same ensemble.

## Experiment 9: full-data MuRIL + TF-IDF ensemble, 2026-09-18

Run 9 is the corrected final-fit follow-up to Run 8. The notebook trains demojized
TF-IDF/SVM and demojized MuRIL once each on all 6,401 deduplicated Task A rows. It does
not perform a holdout or OOF pass, and it does not use the hidden validation labels to
fit weights. It applies the fixed weights learned in Run 8: 57% SVM and 43% MuRIL.

The notebook wrote `task_a_full_ensemble.zip`, containing one bare `predictions.csv`,
and preserved the component probabilities, blend weights, and logs in
`task_a_full_ensemble_outputs`. CodaBench returned **`0.8187` macro-F1** and **`0.8189`
accuracy**, improving on the previous MuRIL result (`0.8163`/`0.8164`) by `+0.0024` and
`+0.0025`. This is now the best recorded Task A result, although the gain is small on
the 806-row validation set.

---

## Experiment 10: RBF SVM on frozen MuRIL embeddings, 2026-09-19

[The notebook](../notebooks/task_a/10_frozen_embeddings_svm.ipynb) uses MuRIL as a feature
extractor with **no fine-tuning at all**: meanmax-pooled embeddings feed an
`SVC(kernel="rbf", C=2.0, class_weight="balanced")`. It scores the method on the fixed
85/15 holdout, then refits the SVM on all 6,401 deduplicated rows and predicts the
official validation inputs.

The artifact is preserved at
[`submissions/task_a_embeddings_svm/`](../submissions/task_a_embeddings_svm/) and was
validated on 2026-09-19: 806 rows, ids matching `binary_validation_inputs.csv` exactly, no
duplicates, `id,label` header, allowed labels only, and a flat ZIP containing one bare
`predictions.csv`.

| | this run | training prior | previous Task A submission |
|---|---|---|---|
| Hate | 55.2% | 49.1% | 49.3% |
| Non-Hate | 44.8% | 50.9% | 50.7% |

It leans about six points further toward `Hate` than either reference. Not a collapse, but
on a balanced task a skewed prior costs macro-F1 on the under-called class.

**CodaBench returned `0.71` macro-F1 and `0.70` accuracy.** That is 11 points below the
current best and 10 below the TF-IDF floor, so freezing the encoder costs far more than an
RBF head recovers. The method is rejected.

**It is also rejected as an ensemble member, and the arithmetic settles that without the
hidden labels.** It disagrees with the best submission on 22.8% of rows, which is normally
the property that makes a blend worth trying. But on a binary task, when two systems
disagree exactly one of them matches gold, so their hit-rates on the disagreement rows sum
to 1. With agreement 0.772 and accuracies 0.8189 and 0.70:

| quantity | value |
|---|---|
| rows where they disagree | 184 of 806 |
| on those rows, the best submission is right | 76.0% |
| on those rows, this model is right | **24.0%** |
| on rows where they agree, both are right | 83.6% |

A blend helps only by overruling the stronger member, and here it would be wrong three
times in four when it did. No weight gains. It should not enter the blend search in
[Run 14](../notebooks/task_a/14_third_member_blend.ipynb).

**The by-product is the more useful finding.** On the 622 rows where the two agree, they
are still wrong 16.4% of the time. Two systems with very different inductive biases, a
fine-tuned transformer and a frozen encoder with a kernel head, failing together on the
same rows matches the Task A error analysis, which found 92% of errors in comments
carrying no profanity at all. That shared residue, not member diversity, is where the
remaining points are. It also lowers the expected value of Run 14: if a frozen MuRIL
disagrees this much and still adds nothing, XLM-R may not either.

The notebook's holdout macro-F1 was not downloaded. Retrieving it is now optional, since
the CodaBench score settles the method.

## Experiment 11: one reinitialized layer and a refitted blend, 2026-09-19

[The notebook](../notebooks/task_a/11_reinit1_full_data.ipynb) changed three things about
the current best Task A recipe and measured them together.

| setting | Run 9, previous best | Run 11 |
|---|---|---|
| SVM | demojized, `--full-fit`, all 6,401 rows | same |
| MuRIL | demojized, `--folds 1`, 1 seed, 6 epochs, effective batch 16 | same |
| reinitialized layers | 2 | **1** |
| blend weight | 0.57 / 0.43, fitted in Run 8 against a two-layer MuRIL | **refitted** |
| decision threshold | 0.5, never tuned | **fitted** |

The weight was refitted rather than carried over because it was fitted against the
component this run replaces: a stronger MuRIL deserves more of the blend. That required
out-of-fold probabilities, so stage 1 ran five folds for both components, stage 2 fitted
the weight and threshold with a nested check, and stage 3 refitted both on all 6,401 rows.
About 3.4 hours.

### Result

CodaBench returned **`0.8188` macro-F1 and `0.8189` accuracy**, against Run 9's `0.8187`
and `0.8189`. That is `+0.0001`, less than one row of 806, with accuracy identical to four
decimals. **A tie.**

Three changes at once moved nothing. Two readings fit that, and the stage-2 output
distinguishes them:

* if the refitted weight came out near 0.57, one layer did not make MuRIL stronger, and
  Task B's finding that one beat two by +0.5 does not transfer to Task A
* if the weight moved a long way and the score still did not budge, the blend is flat in
  that region and the weight was never the constraint

The stage-2 numbers — both components' OOF scores, the nested blend score, the fitted
weight and threshold — are **not yet recorded**. They are the informative part of this run,
measured on 6,401 rows at about 0.6 points of noise against the roughly 1.5 points one
CodaBench score carries. Retrieve them from the notebook output.

**Either recipe can serve as the base**, since they are indistinguishable. The durable
product is the stored `oof_probs.npy` for both components, which Task A had never had:
every later blend weight, threshold or decode rule is now seconds of CPU rather than a GPU
session.

## Experiment 16: the funnel over component-level ideas, created 2026-09-19

[The notebook](../notebooks/task_a/16_funnel.ipynb) and
[`experiments/task_a/funnel.py`](../experiments/task_a/funnel.py) replace Runs 12 to 15 as
the way to answer "what helps".

Eight full-data submissions cannot answer it: one CodaBench score on 806 rows carries about
1.5 points of standard deviation and most of these ideas are worth one or two. Eight
five-fold runs would answer it, at ~160 minutes each, about 30 hours, and would still never
put the factors on one table.

So: screen every arm on the fixed 15% holdout at ~33 minutes, then promote only the
leaders. Task B ran this funnel and recorded that the holdout ranked all four promoted arms
in the same order five folds did, while reading about 0.022 high. Stage 1 therefore orders
arms and never reports a number.

| arm | the change | prior evidence |
|---|---|---|
| `control` | one reinitialized layer, 6 epochs | the reference |
| `reinit2` | two layers | Run 9's inherited setting |
| `tapt` | TAPT checkpoint | +2.9 on Task B, the largest effect measured anywhere here |
| `ext_mix` | external rows in training | +0.0030 on the TF-IDF floor |
| `ext_stage` | external first, then Task A | lets clean labels overwrite the external boundary |
| `epochs10` | 10 epochs | Task A logs show validation F1 still rising at epoch 6 |
| `large` | MuRIL-large, `--no-fgm`, `--lr 1e-5` | untested anywhere |

Every arm changes the MuRIL component only; the SVM half of the blend is fixed because
nothing queued touches it. That is also the dependency order: a blend weight depends on how
strong its components are, so the component must be settled before Run 11 fits the weight.

Two safeguards. Each arm's predicted positive rate is checked against Task A's 0.491 prior,
and an arm calling one class on over 90% of rows is not promoted — which is what mDeBERTa
did on Task B at a learning rate that suited the base model. And every stage skips work
whose output exists, so a session that dies partway resumes rather than restarts.

Record the whole table here when it runs, **including the arms that lose**.

## Experiment 12: does TAPT help Task A? 2026-09-20

[The notebook](../notebooks/task_a/12_tapt_oof.ipynb) trained two five-fold arms differing
only in the starting encoder, both with one reinitialized layer, six epochs and seed 42.
The TAPT corpus was 9,693 unique comments: Task A's own text plus the external Kannada
corpus, no labels read from either.

| arm | OOF macro-F1 |
|---|---|
| `task_a_tapt_5f` | **0.8128** |
| `task_a_stock_5f` | 0.7894 |
| **TAPT effect** | **+0.0234** |

Fold noise on 6,401 rows is about 0.006, so +0.0234 is roughly four times it. **TAPT
transfers to Task A.** Task B measured +0.0286 under the same convention, so the two agree
closely.

It also clears the TF-IDF floor: TAPT MuRIL alone scores 0.8128 against the floor's 0.8073,
where stock MuRIL at 0.7894 does not. This is the first Task A encoder to beat the floor on
identical folds.

**The stated caveat still applies.** The MLM stage read every Task A training comment,
including the rows each fold later scored itself on. No labels were read, but the TAPT arm
saw the wording of its own out-of-fold rows and the control did not, so the comparison is
biased in TAPT's favour. At +0.0234 the bias is unlikely to account for all of it.
[Run 7](../notebooks/task_a/07_tapt_holdout.ipynb) is the leak-free version and is now
worth running to bound it.

## Experiment 16: the funnel, stage 1 results, 2026-09-20

Six arms screened on the fixed 15% holdout, 960 rows, 3.6 hours. Noise on that split is
about 0.013, so treat anything smaller as unresolved. All six predicted positive rates
landed between 0.42 and 0.48 against a 0.491 prior: **nothing collapsed, including
MuRIL-large.**

| arm | holdout macro-F1 | vs control | read |
|---|---|---|---|
| `epochs10` | **0.8103** | +0.0283 | real, and the largest |
| `large` | **0.8029** | +0.0209 | real; MuRIL-large did not collapse |
| `reinit2` | **0.7980** | +0.0160 | real |
| `control` | 0.7820 | — | one layer, six epochs |
| `ext_mix` | 0.7798 | -0.0022 | noise; dead |
| `ext_stage` | 0.7754 | -0.0066 | noise; dead |

### Three conclusions

**Ten epochs is the biggest single lever found on Task A.** The prediction from the epoch
curve was "a few tenths"; the measurement is +2.8 points. Six epochs was badly
undertrained, and the extrapolation understated it because a longer cosine schedule changes
the shape of training rather than merely extending it.

**Two reinitialized layers beats one, by +1.6.** This is the opposite of Task B, where one
beat none by +3.0 and one beat two by +0.5. It also explains Run 11's tie: changing 2 to 1
was neutral-to-negative, and the refitted weight and threshold absorbed the difference.
**Task A should keep `--reinit-layers 2`.**

**The external corpus's labels do not help Task A.** Both arms land inside noise and both
are negative. The one opening Task A had that Task B did not is closed. The rules question
attached to training on external labels no longer needs answering.

### Bug found and fixed

`muril.py` never printed a holdout macro-F1 for `--folds 0`; it only saved
`holdout_probs.npy`. `funnel.py` parsed for a line that did not exist, so every arm came
back as a dash and the table it wrote was empty. The scores above were recovered from the
per-fold `[s42holdout] best ... last ...` lines the trainer has always logged. Both sides
are fixed: `muril.py` now prints the score, and `funnel.py` has a fallback that reads the
older form.

## Task B

The completed Kaggle runs and follow-up ablations are listed below. Local figures are the
unbiased `last`-checkpoint number; CodaBench figures are from the 395-row validation phase
with hidden labels. The current status and ordered roadmap are at the top of this document.

### CodaBench validation scores

| submission | run | recipe | macro-F1 |
|---|---|---|---|
| `f_tapt` | 2 | TAPT MuRIL, 5 seeds on all 3,143 deduplicated rows | 0.6007¹ |
| `b_tapt_5f` | 1 | TAPT MuRIL, five fold models averaged | 0.5922 |
| `d0v0_noaux_full` | 4 | TAPT MuRIL, 5 seeds on all 3,159 rows, TAPT on every comment | not run yet |
| `b_reinit1_full` | 8 | TAPT MuRIL, one-layer reinit, 5 seeds on all 3,159 rows | **0.6299** |
| `b_reinit1_rdrop_full` | 9 | TAPT MuRIL, one-layer reinit + R-Drop 0.5, 5 seeds on all 3,159 rows | **0.6410** |

¹ Inferred, not confirmed: the `scoring_result.zip` reading 0.6007 was downloaded a few
minutes after `f_tapt.zip`. Check the CodaBench submission list.

One CodaBench score carries about 3.0 points of standard deviation (bootstrap of
macro-F1 at 395 rows), so the 0.85-point gap between these two is not evidence that
either recipe is better.

### Run 1 -- overnight sweep, 2026-09-12

`experiments/task_b/overnight.py` via `01_baseline_sweep.ipynb`, 8.44 h. Encoder and loss ideas ranked
against stock MuRIL on a 15% holdout, the best four promoted to five folds.

Five-fold runs, on all 3,143 rows:

| run | macro-F1 | note |
|---|---|---|
| `b_tapt_5f` | **0.6013** | domain-adapted MuRIL -- **submitted**, 0.5922 on CodaBench |
| `ensemble_b` | 0.6002 | weight-searched blend, nested estimate |
| `svm_b` | 0.5948 | TF-IDF + LinearSVC floor |
| `b_base_5f` | 0.5727 | stock MuRIL, the reference |
| `b_rdrop_5f` | 0.5614 | R-Drop 0.5 |
| `b_focal_5f` | 0.5572 | focal loss, gamma 2.0 |

Holdout arms, 15% of rows (472):

| run | macro-F1 | note |
|---|---|---|
| `b_tapt` | 0.5972 | domain-adapted MuRIL |
| `b_base` | 0.5948 | stock MuRIL |
| `b_rdrop` | 0.5845 | R-Drop |
| `b_focal` | 0.5779 | focal loss |
| `b_abusive` | 0.5718 | abusive-tuned MuRIL warm start |
| `b_hing` | 0.5678 | HingRoBERTa |
| `b_xlmr` | 0.5561 | XLM-R base |
| `b_mdeberta` | 0.1002 | collapsed; 3e-5 is too high for mDeBERTa |
| `b_large` | -- | MuRIL-large, cut by the budget guard |

* **Only task-adaptive pretraining helped.** +2.9 points over stock MuRIL on
  identical folds. Nothing else beat the reference.
* **Holdout ranks correctly but reads about two points high.** For all four
  promoted arms the five-fold order matched the holdout order exactly, while
  the level dropped by roughly 0.022 in three of the four cases.
* **Blending did not pay.** The nested estimate landed below the best single
  model, so `ensemble.py` recommended the single run and that is what was sent.
* **The macro-F1-optimal decode did not pay either.** `src/hastika/task_b/decode.py` fits
  per-class decision weights on the OOF and validates them nested; it lost on
  every run and wrote nothing. On the floor it scored 0.5856 against 0.5948 for
  plain argmax.
* **Prediction distribution is the sanity check that matters.** The floor called
  `Violence` on 6 of 395 test rows against a 7.0% training rate, which under
  macro-F1 forfeits most of a class. The submitted run calls it 24 times.

### Run 2 -- five-seed full fits, 2026-09-13

`experiments/task_b/fullfit.py` via `02_fullfit_sweep.ipynb`, 9.35 h. Six ideas, each trained
five times on all 3,143 deduplicated rows with the probabilities averaged. TAPT still
held 5% of its text back and dropped one-word comments, so it used 6,044 of 6,406.

**No local score exists for these and none can.** The `holdout` column is the Run 1
holdout score of the same idea, copied in for ranking, not a measurement of these models.

| arm | holdout (Run 1) | agrees with `b_tapt_5f` | largest drift from prior | note |
|---|---|---|---|---|
| `f_tapt` | 0.5972 | 91.1% | 3.2 points | MLM-adapted MuRIL -- submitted, 0.6007 on CodaBench¹ |
| `f_base` | 0.5948 | 87.1% | 2.7 | stock MuRIL |
| `f_rdrop` | 0.5845 | 85.8% | 3.0 | R-Drop 0.5 |
| `f_focal` | 0.5779 | 85.1% | 4.0 | focal loss, gamma 2 |
| `f_abusive` | 0.5718 | 81.3% | 3.0 | abusive-tuned warm start |
| `f_hing` | 0.5678 | 81.0% | 3.9 | HingRoBERTa |

### Run 3 -- data-processing grid, 2026-09-13 to 14

`experiments/task_b/grid.py` via `03_factorial_grid.ipynb`, 8.87 h. Every cell is a 15% holdout run on
the same 472 rows, so these are real scores.

* `corpus` D0 = TAPT on Kannada text only; D1 = plus Tamil and Malayalam
  OffensEval-Dravidian, about 58,000 comments, one epoch so the compute matches.
* `vocab` V0 = MuRIL's tokenizer as shipped; V1 = extended with frequent word-forms.
* `aux` = auxiliary head predicting violent language, weight 0.3.

| config | corpus | vocab | aux | seeds | macro-F1 | vs stock MuRIL |
|---|---|---|---|---|---|---|
| `D0_V0_noaux` | D0 | V0 | no | 1 | **0.5972** | +0.0024 |
| `stock_V0_noaux` | - | V0 | no | 1 | 0.5948 | +0.0000 |
| `D0_V0_noaux_multiseed` | D0 | V0 | no | 3 | 0.5917 | -0.0031 |
| `D0_V0_aux` | D0 | V0 | yes | 1 | 0.5916 | -0.0032 |
| `D1_V0_aux` | D1 | V0 | yes | 1 | 0.5875 | -0.0073 |
| `D1_V0_noaux` | D1 | V0 | no | 1 | 0.5865 | -0.0083 |
| `stock_V0_noaux_multiseed` | - | V0 | no | 3 | 0.5820 | -0.0128 |
| `D0_V1_noaux` | D0 | V1 | no | 1 | 0.5656 | -0.0292 |
| `D0_V1_aux` | D0 | V1 | yes | 1 | 0.5546 | -0.0402 |
| `stock_V0_aux` | - | V0 | yes | 1 | 0.5495 | -0.0453 |
| `D1_V1_noaux` | D1 | V1 | no | 1 | 0.5117 | -0.0831 |
| `D1_V1_aux` | D1 | V1 | yes | 1 | 0.5071 | -0.0877 |

* **None of the three levers helped.** `D0_V0_noaux` is the same recipe as `b_tapt`
  and reproduced its 0.5972 exactly.
* **The top seven cells are one result.** The holdout's macro-F1 has a bootstrap
  standard deviation of 2.7 points and those cells span 1.5.
* **Vocabulary extension is the only real effect, and it hurts:** 3 to 8 points below
  the same cell with V0. New embeddings, initialized as the mean of their pieces, do
  not get enough training from 3k comments.
* **Three seeds scored below seed 42 alone** for both stock and TAPT, so seed 42 was
  a lucky draw on this split.
* **The grid is tilted toward TAPT.** TAPT read every training comment's text,
  including the 472 holdout rows (never their labels). The case for TAPT rests on
  Run 1's five-fold comparison instead.
* **`subs/D0_V0_noaux.zip` is not a submission candidate.** It is the single-seed
  holdout model trained on 85% of the rows. The grid's own five-seed full fit only
  starts with about two hours of budget left and the run ended at 8.87 of 10.25 h, so
  it most likely skipped; check the Kaggle output for `subs/winner_full.zip`.

### Run 4 -- `D0_V0_noaux` on 100% of the data, not run yet

`notebooks/task_b/04_full_data_fit.ipynb`, about 2 h. The Run 3 winner with nothing held back
at either stage:

| stage | `f_tapt` (Run 2) | Run 4 |
|---|---|---|
| TAPT text | 6,044 of 6,406 (94.3%) | **6,406 of 6,406** |
| classifier rows per model | 3,143 of 3,159 (deduplicated) | **3,159 of 3,159** |
| seeds | 42-46 | 42-46 |

Its only score will be CodaBench's. Record it in the table at the top of this section.

### Run 5 -- one-layer reinitialization ablation, 2026-09-16

`notebooks/task_b/05_reinit_one_layer.ipynb` tests whether reinitializing only the final
MuRIL encoder layer is better than the established two-layer reset. The notebook builds
one shared TAPT checkpoint, then runs both classifier variants with the same deduplicated
Task B data, five-fold split, split seed 42, model seed 42, six epochs, all existing
regularization, no auxiliary head, and `--select last`. This isolates the
`--reinit-layers` choice:

| run | reinitialized layers | status |
|---|---:|---|
| `b_tapt_reinit2_5f` | 2 (control) | 0.6013 OOF macro-F1 |
| `b_tapt_reinit1_5f` | 1 (ablation) | **0.6102 OOF macro-F1** |

The run used one shared TAPT checkpoint (`6044` of `6362` cleaned comments, with `318`
held out for MLM perplexity), the deduplicated 3,143-row Task B set, five folds, seed 42,
six epochs, the existing regularization, no auxiliary head, and `--select last`. The
one-layer variant improved macro-F1 by `+0.0089` and accuracy from `0.6557` to `0.6592`.
Per-class F1 changed as follows:

| class | two layers | one layer | delta |
|---|---:|---:|---:|
| Gender | 0.735 | 0.730 | -0.005 |
| Geo-political | 0.592 | 0.590 | -0.002 |
| Others | 0.454 | 0.476 | +0.022 |
| Political | 0.783 | 0.797 | +0.014 |
| Religion | 0.716 | 0.717 | +0.001 |
| Violence | 0.328 | 0.352 | +0.024 |

This is encouraging but is still one five-fold run with model seed 42. The notebook's
independent comparison cell failed after training because the live kernel did not refresh
`PYTHONPATH`; the training logs produced the scores above. The source notebook now inserts
`src/` into the live kernel path. Confirm the one-layer result with seeds 43 and 44 before
changing the recommended recipe. There is no CodaBench submission in this experiment.

### Run 6 -- confirm one-layer reinitialization, 2026-09-17

`notebooks/task_b/06_reinit_confirmation.ipynb` repeats Run 5 at model seeds 43 and 44.
It builds one shared TAPT checkpoint and runs four separate five-fold comparisons:
The captured run log is [run06_reinit_confirmation.log](../results/task_b/logs/run06_reinit_confirmation.log).

| run | reinitialized layers | model seed | status |
|---|---:|---:|---|
| `b_tapt_reinit2_s43` | 2 (control) | 43 | `0.6026` OOF macro-F1 |
| `b_tapt_reinit1_s43` | 1 (ablation) | 43 | **`0.6127`** OOF macro-F1 |
| `b_tapt_reinit2_s44` | 2 (control) | 44 | `0.6083` OOF macro-F1 |
| `b_tapt_reinit1_s44` | 1 (ablation) | 44 | `0.6082` OOF macro-F1 |

All settings other than `--reinit-layers` match Run 5. The notebook reports each seed's
OOF macro-F1, the mean of the two seed scores, and macro-F1 after averaging the two seeds'
OOF probabilities. It also prints per-class reports. The one-layer mean seed score was
`0.6105` versus `0.6055` for two layers; averaged probabilities scored `0.614` versus
`0.612`. The small positive aggregate supports retaining one-layer reinitialization.
There is no submission in this run.

### Run 7 -- no encoder-layer reinitialization ablation, 2026-09-17

`notebooks/task_b/07_no_reinit_ablation.ipynb` tests whether preserving the final encoder
layer as well is better than the current one-layer setting. `--reinit-layers 1` is the
control; `--reinit-layers 0` keeps both layers 11 and 12 from the TAPT checkpoint:
The captured run log is [run07_no_reinit_ablation.log](../results/task_b/logs/run07_no_reinit_ablation.log).

| run | reinitialized layers | model seed | status |
|---|---:|---:|---|
| `b_tapt_reinit1_vs0_s43` | 1 (control) | 43 | `0.6127` OOF macro-F1 |
| `b_tapt_reinit0_s43` | 0 (ablation) | 43 | `0.5854` OOF macro-F1 |
| `b_tapt_reinit1_vs0_s44` | 1 (control) | 44 | `0.6082` OOF macro-F1 |
| `b_tapt_reinit0_s44` | 0 (ablation) | 44 | `0.5760` OOF macro-F1 |

All other settings match Runs 5 and 6: one shared TAPT checkpoint, deduplicated Task B
data, fixed five-fold split with split seed 42, six epochs, existing regularization, no
auxiliary head, and `--select last`. The notebook reports each seed, the mean seed score,
the score after averaging OOF probabilities, and per-class F1. One-layer averaged OOF
macro-F1 was `0.614` versus `0.584` with no reinitialization, so the zero-layer option is
discarded. There is no submission in this run.

### Run 8 -- full-data one-layer fit and official validation inference, 2026-09-17

`notebooks/task_b/08_full_data_fit_reinit1.ipynb` is the provisional final-fit notebook
for the currently best observed recipe. It uses `--reinit-layers 1`, TAPT on all 6,406
allowed comments, and five classifier seeds (`42--46`) trained on all 3,159 labelled Task
B rows. It then averages the five validation probability matrices and predicts all 395
rows in `data/raw/multiclass_validation_inputs.csv`.

| output | purpose | status |
|---|---|---|
| `b_reinit1_full` | five-seed full-data predictions and probabilities | completed |
| `b_reinit1_full.zip` | validated `id,label` payload for CodaBench | **0.6299** |

The captured log is [run08_full_data_reinit1.log](../results/task_b/logs/run08_full_data_reinit1.log).
It confirms TAPT on all 6,406 comments and five classifier seeds (`42--46`) on all 3,159
labelled rows with `--reinit-layers 1`. The submission helper
validated 395 rows and a ZIP containing one bare `predictions.csv`. CodaBench returned
macro-F1 `0.6299` and accuracy `0.6886`; Run 9 subsequently surpassed this result.

The notebook's final diagnostic cell raised `NameError: pd is not defined` after the ZIP had
already been written. This affected only the distribution-reporting cell, not training,
prediction generation, ZIP validation, or the submitted result; the canonical notebook now
imports pandas before that cell.

### Run 9 -- full-data R-Drop submission, 2026-09-17

`notebooks/task_b/09_rdrop_one_layer.ipynb` trains the R-Drop version of the current
one-layer TAPT MuRIL recipe directly on all labelled Task B rows. It uses TAPT on all
6,406 permitted comments, `--reinit-layers 1`, `--rdrop 0.5`, five seeds (`42--46`), six
epochs, balanced class weighting, FGM, EMA, and no auxiliary head. The five validation
probability matrices are averaged. This is a final-fit submission run, so it has no local
OOF score and does not include a no-R-Drop control.

The notebook validated the 395-row prediction file and wrote the single candidate
`b_reinit1_rdrop_full.zip`, containing one bare `predictions.csv`. CodaBench returned
macro-F1 **`0.6410`** and accuracy **`0.6937`**, improving on Run 8's `0.6299` macro-F1
and `0.6886` accuracy by `+0.0111` and `+0.0051`, respectively. This is now the best
confirmed Task B result. The captured log is [run09_rdrop_full.log](../results/task_b/logs/run09_rdrop_full.log).
