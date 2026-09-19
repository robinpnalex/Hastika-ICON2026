# Ideas

Every idea considered for either task, with the evidence for and against it. Completed
experiments live in [`EXPERIMENTS.md`](EXPERIMENTS.md); this file is the backlog and the
graveyard, so an idea that was measured and lost stays written down rather than being
rediscovered in three weeks.

## The standing constraint

From 2026-09-19, every new experiment is:

* **stacked on the current best submission**, one change at a time
* trained on **100% of the labelled rows**, `--folds 1`, no holdout and no OOF pass
* scored **only by CodaBench**

That is a deliberate trade and it has a cost worth stating once. A full-data fit has no
local score, so the 806-row validation set is the only readout, and one CodaBench score
carries roughly 1.5 points of standard deviation on Task A. A gap under about 3 points
between two submissions is not evidence. Ideas below are therefore ranked by expected
effect size, and the ones expected to move less than a point are marked as such: they are
worth stacking into a recipe, not worth a submission slot of their own.

## The current best recipes

**Task A — 0.8187 macro-F1, 0.8189 accuracy.** A fixed blend of two components, each
fitted on all 6,401 deduplicated rows:

| component | recipe | weight |
|---|---|---|
| TF-IDF / LinearSVC | word 1-2 grams + char_wb 2-5 grams, calibrated, demojized, `--full-fit` | 0.57 |
| MuRIL | demojized, `--folds 1`, **one seed**, 6 epochs, **two** reinitialized layers, effective batch 16 | 0.43 |

Decision threshold 0.5. The weights came from a five-fold search in Run 8.

**Task B — 0.6410 macro-F1, 0.6937 accuracy.** `b_reinit1_rdrop_full`: TAPT MuRIL, **one**
reinitialized layer, R-Drop 0.5, five seeds on all 3,159 rows, no deduplication.

## Two rules that hold across both tasks

**1. Representation-level changes work; input features do not.** TAPT (+2.9 on Task B) and
layer reinitialization (+3.0) are the only things that ever beat a reference. Gazetteer
tags, an auxiliary head, vocabulary extension, stemming, stopword removal, synthetic data
and per-class decode weights all measured inside noise or worse.

**2. Whatever is lexically obvious, the model already learned.** Measured twice. On Task B,
entity gazetteers fire almost entirely on the three classes already solved: Religion 63%,
Political 52%, Geo-political 48% of rows carry a signature entity, against 19%, 19% and 18%
for Gender, Others and Violence. On Task A, **92% of all errors are in comments containing
no profanity at all**; the 18% of rows that do contain a slur are already 91.6% accurate.
Any feature built from a word list targets rows that are already right.

---

## Ranked backlog

| # | idea | task | expected | cost | status |
|---|---|---|---|---|---|
| 1 | TAPT | A | +0.5 to +2.0 | 6 h | notebook written |
| 2 | External corpus **labels** | A | +0.3 to +1.0 | 4 h full-data | notebook written, 5-fold |
| 3 | One reinitialized layer instead of two | A | +0.0 to +0.5 | **40 min** | Run 11, full data, ready |
| 4 | Five seeds instead of one | A | +0.2 to +0.5, cannot hurt | 3 h | in Run 5 |
| 5 | Tuned decision threshold on the blend | A | +0.35 measured | free, needs OOF | not built |
| 6 | Refit the blend weight | A | small, unknown sign | free, needs OOF | not built |
| 7 | R-Drop | A | unknown | 6.5 h | notebook written |
| 8 | Ten epochs instead of six | A | +0.1 to +0.4 | 4.5 h | notebook written |
| 9 | MuRIL-large | A, B | -8 to +1.5 | 8 h | notebook written |
| 10 | Whole-word masking in TAPT | B | +0.3 to +1.0 | 2 h | not built |
| 11 | Self-distillation from OOF soft labels | B | +0.3 to +1.0 | 4 h | not built |
| 12 | Ensemble of independently TAPT'd encoders | B | +0.3 to +0.8 | 6 h | not built |
| 13 | Context-conditional decode | B | +0 to +1.9 | 2.6 h | notebook written |
| 14 | Third ensemble member, XLM-R | A | +0 to +0.5 | 6 h | notebook written, downgraded |
| 15 | Pseudo-labelling the validation inputs | A, B | +0 to +0.3 | 2 h | not built |
| 16 | Supervised contrastive auxiliary loss | A, B | unknown | 4 h | not built |

### 1. TAPT for Task A

Continue MuRIL's masked language modelling on Task A's own comments plus the external
Kannada text, then fine-tune. Worth **+2.9 points** on Task B, 0.6013 against 0.5727 on
identical folds, and the only intervention of about a dozen there that beat the reference.
Task A has never tried it.

Two reasons it may transfer less: Task A has twice the supervised data, so the encoder
needs adaptation less, and its comments are the same register Task B's came from, so the
adaptation is not new information about the domain. Two reasons it may transfer more: more
in-domain text to adapt on, and a binary target that is easier to fit once the
representation is right.

Stack: replace `google/muril-base-cased` with the TAPT checkpoint in the MuRIL component.

### 2. The external corpus's labels

`data/external/offenseval_kn.csv` carries `Hate` / `Non-Hate`, which is **exactly** Task
A's label space. On Task B the six-way taxonomy made those labels unusable and only the
text was ever used, in TAPT. This is the one opening Task A has that Task B did not.

Measured on the TF-IDF floor, five-fold: `0.8073` to `0.8103`. It adds 3,247 rows to
6,401, a 51% increase, and transformers gain more from extra data than linear models do.

Two modes, both already implemented as `--external-mode mix` and `stage`. `mix` appends
the rows to training; `stage` fine-tunes on them first and then on Task A, so a second
pass on clean labels overwrites whatever boundary the external annotation set.

**Rules note.** This trains on external *labels*, not just text. TAPT deliberately avoids
that and every Task B decision kept external labels out. Settle it before submitting.

### 3. One reinitialized layer instead of two — Run 11, ready

**This is the one to run first.** Forty minutes, and it decides the base recipe that every
other idea stacks on.


Task B measured one layer against none at **+3.0**, and one against two at +0.5, which is
inside noise. Task A's best submission uses two, inherited rather than chosen. The
reinitialization itself is clearly worth having; the count is a coin flip that Task A has
never called for itself.

### 4. Five seeds instead of one

Task A's best MuRIL component is a **single seed**. Averaging five costs nothing but time
and cannot make the model worse in expectation. This is the same move that took Task B from
0.5922 to 0.6007 when folds became full-data seeds.

Seeds vary three things here: the random weights of the reinitialized layers and the head,
the batch shuffle order, and the dropout masks. The data is identical.

### 5. Tuned decision threshold

Measured today on the TF-IDF floor: threshold 0.5 gives `0.8073`, a nested threshold gives
`0.8108`, **+0.0035 for no GPU at all**. `hastika.models.muril` already does this for a
single model and applies it only when the gain clears 0.002, but the ensemble notebook
hard-codes `> 0.5`, so the blend has never had it.

On a balanced binary task the macro-F1-optimal threshold is usually near 0.5 but not on it,
and the blend of two calibrated components has no reason to land exactly there.

### 6. Refit the blend weight

The 57/43 split was fitted in Run 8 against a **two-layer, single-seed** MuRIL. Any change
to the MuRIL component changes the weight it deserves. Reinitialization cannot interfere
with the ensemble — it touches only MuRIL's top layers and the SVM is untouched — but it
does move the optimum.

Free, given out-of-fold probabilities for both components — which no Task A run has yet
stored, so ideas 5 and 6 are both blocked on the one five-fold pass described at the end of
this file. Run 11 deliberately holds the weights and the threshold fixed so its result is
attributable to the reinitialization alone.

### 7. R-Drop

Two stochastic forward passes per step tied by a symmetric KL penalty. It is in Task B's
winning recipe, where it is credited with +1.1 on the leaderboard, but that gap is a third
of a 395-row set's standard deviation and the only *local* measurement of R-Drop, on a
different encoder, was 4 points negative. Genuinely unresolved on either task.

### 8. Ten epochs instead of six

Read out of the existing Task A logs, averaged over 14 fold-runs: validation macro-F1 is
still rising at epoch 6.

| epoch | mean val macro-F1 | gain |
|---|---|---|
| 4 | 0.7852 | +0.0107 |
| 5 | 0.7906 | +0.0054 |
| 6 | 0.7944 | +0.0038 |

Six is a cutoff, not a converged point, and Task B showed the same shape with 13 of 20
folds peaking at the last epoch. But the gains halve each epoch, so extrapolation gives
only a few tenths. The cosine schedule is the wildcard: at ten epochs the learning rate
decays more slowly, so the shape of training changes rather than simply extending.

Also relevant: Task A's 6,401 rows give about 2,400 optimizer steps in six epochs at
effective batch 16; Task B's 3,159 rows give about 1,180. The epoch count was tuned on the
larger of the two.

### 9. MuRIL-large

24 layers against 12, never tried on either task. Task A's 6,401 rows support a larger
model far better than Task B's 3,143. Needs `--no-fgm`, because FGM clones the embedding
table every step, and `--lr 1e-5`, because 3e-5 is exactly what collapsed mDeBERTa to
0.1002 on Task B.

High variance in both directions. Check predicted class balance on one fold before
spending a full run on it.

### 10. Whole-word masking in TAPT

`tapt.py` masks individual wordpieces at random, and MuRIL splits each word here into about
2.3 pieces, so the model often fills in a piece while the rest of its own word is visible.
That teaches spelling completion rather than context. Masking whole words forces
reconstruction from surroundings, which is what a corpus with 35% rare-token occurrences
needs. A change to the masking collator plus one TAPT run.

### 11. Self-distillation

Train on targets mixing the gold label with out-of-fold probabilities from a five-fold run
of the same recipe. On Task B a third of all errors sit on the Gender/Others boundary,
which a dedicated two-class model could not separate (-0.003), so it behaves like label
noise; the training data itself contains comments annotated both ways. Soft targets stop
the model chasing one annotator's coin flip. Uses only training labels.

### 12. Ensemble of independently TAPT'd encoders

The only real gains came at the representation level, so diversity should come from there
too, rather than from more fine-tuning seeds on one encoder. Two or three TAPT runs with
different seeds or masking schemes, each fine-tuned, then averaged.

### 13. Context-conditional decode, Task B

A violent action word means different things in different company. Among the 420 training
comments containing one: with media context present, Others is 0.27 and Violence 0.18; with
no target named, Violence is 0.34 and Others 0.06. `context_decode.py` learns a lift per
cell and adds it to the log probabilities after training. On the calibrated TF-IDF SVM it
is worth +2.9 points nested. Whether MuRIL already represents the interaction is the open
question. Costs no GPU once a five-fold run exists.

### 14. Third ensemble member — downgraded

Blending pays when members fail differently. Run 10 tested that directly and the answer was
discouraging: a frozen MuRIL with an RBF head disagrees with the best submission on 22.8%
of rows, and on exactly those rows it is right only **24%** of the time. No weight gains.
Worse, on the 622 rows where the two agree they are still wrong 16.4% of the time, so two
very different inductive biases fail on the same comments.

XLM-R may be a better member than a frozen encoder, but the prior on member diversity as a
lever dropped sharply.

### 15. Pseudo-labelling the validation inputs

Transductive, and permitted under the stated rules, which allow validation text without
labels. Add high-confidence predictions on the 806 unlabelled inputs to training. Small n,
and the risk is reinforcing existing errors.

### 16. Supervised contrastive auxiliary loss

Pull same-class comments together in representation space regardless of surface overlap.
The motivation is that 72% of word types are hapax and a held-out comment's nearest
training neighbour has a median cosine similarity of 0.27, so surface matching genuinely
fails. Lowest confidence of anything on this list.

---

## Measured and dead

Do not rebuild these. Each cost a run and each is written down so it does not cost another.

| idea | result | where |
|---|---|---|
| Frozen MuRIL embeddings + RBF SVM | 0.71 on Task A, 11 points below best; also useless in a blend | Run 10 |
| Vocabulary extension for TAPT | -3 to -9 points, the only clearly real effect in the factorial grid | Task B Run 3 |
| Auxiliary head on the act axis | -0.6 with TAPT, -4.5 on stock | Task B Run 3 |
| Focal loss | 0.5572 against 0.6013 on five folds | Task B Run 1 |
| Gazetteer / NER / mood / address tags | -0.007 on the SVM, noise | `features.py` |
| Per-class decode weights | lost its nested check on every run ever tried | `decode.py` |
| More Dravidian TAPT text, Tamil and Malayalam | -0.8 to -1.0 | Task B Run 3 |
| Synthetic generated data | real-vs-synthetic AUC 0.942, gain inside noise | `experiments/task_a/synthetic/` |
| Character normalization, transliteration-lite | -0.003 and +0.001 | Task B setup notes |
| Stopword removal | the 60-word frequency stoplist contains `bjp`, `congress`, `dagar`, `desha` | `normalize.py` |
| Stemming | wordpiece already splits `madthare` into stem and suffixes | `normalize.py` |
| Profanity / insult-density features | 92% of Task A errors contain no profanity | measured 2026-09-19 |
| Class weighting on Task A | the task is 3,286 / 3,160 | — |
| MuRIL + TF-IDF OOF blend, Run 8 weights | 0.7890, rejected; the full-data version at Run 9 scored 0.8187 | Task A Runs 8, 9 |
| XLM-R as a standalone Task B encoder | 0.5561 against MuRIL's 0.5948 | Task B Run 1 |
| HingRoBERTa | 0.5678 | Task B Run 1 |
| mDeBERTa | collapsed to 0.1002; 3e-5 is too high for it | Task B Run 1 |

## How the experiments fit together

The ideas above are not independent, and running them as independent submissions would not
tell you which one helped. Two dependencies drive the order:

1. **Component before blend.** A blend weight depends on how strong its components are, so
   the MuRIL component must be settled before the weight is fitted. Fitting the weight and
   then changing the encoder invalidates the weight — which is exactly what happened to
   Run 8's 57/43, fitted against a two-layer MuRIL that Run 11 now changes.
2. **Measure before ship.** Anything worth under 3 points cannot be resolved on 806
   CodaBench rows, but can be on 6,401 out-of-fold rows. So decide locally, submit once.

That gives three phases:

| phase | what | where | cost |
|---|---|---|---|
| screen | seven component-level arms ranked on the 15% holdout | Run 16 stage 1 | ~6 h |
| confirm | the leaders plus the control on five folds | Run 16 stage 2 | ~160 min per arm |
| ship | fit the weight and threshold on the winner's OOF, refit on all rows, submit | Run 11 | ~35 min |

Run 11 doubles as the first pass of this: its five-fold MuRIL arm is the same configuration
as the funnel's `control`, so running it first supplies the funnel's reference point and the
first `oof_probs.npy` Task A has ever had.

By the end there is one table saying what helps and what does not, measured on 6,401 rows,
with one submission spent on the answer rather than eight spent failing to find it.

## The measurement problem, and the cheapest fix

With no holdout, 806 rows decide everything on Task A and 395 on Task B. Most ideas above
are worth under 3 points, which those sets cannot resolve.

The cheapest mitigation costs one run: a single five-fold pass stores `oof_probs.npy` for
the current recipe. Nothing in the repository has one for either task. With that file,
every blend weight, every decision threshold and every decode rule becomes seconds of CPU
instead of hours of GPU, and can be checked honestly with a nested estimate before a
submission slot is spent. It is the highest-leverage hour available, and it is the one
thing the no-holdout constraint should make an exception for.
