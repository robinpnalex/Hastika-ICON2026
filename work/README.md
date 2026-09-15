# HASTIKA working directory

Modelling code for both sub-tasks. Env: `~/.venvs/hastika`, or `uv run` from the
repo root. Task B has its own guide in [`RUN_TASK_B.md`](RUN_TASK_B.md).

## Shared

| File | What it does |
|------|--------------|
| `prep.py` | Text cleaning. The released CSVs are **mojibake** (UTF-8 bytes decoded as latin-1), so Kannada script and emoji arrive mangled. Undoes that, unescapes HTML entities, strips `<br>`, normalizes URLs/@mentions. Also `dedupe_index`, which drops repeated comments and label-conflict groups. |
| `muril.py` | The tuned fine-tuning recipe both tasks share: meanmax pooling, layer-wise LR decay, top-layer re-init, FGM adversarial training, EMA, multi-seed, twice-per-epoch checkpoint selection on macro-F1. Architecture-agnostic — `--model` takes any HF encoder. |
| `ensemble.py` | Blends runs that have OOF matrices, by weight search seeded with the single-model corners. Reports a **nested** estimate and says so when blending does not pay. |
| `make_submission.py` | Validates a `predictions.csv` against the task rules and writes a flat zip. Build every submission with this. |

## Task A

| File | What it does |
|------|--------------|
| `baseline_svm.py` | TF-IDF char(2-5) + word(1-2) n-grams → LinearSVC. 5-fold CV, seconds, no GPU. The floor. `--task b` switches it to six classes. |
| `train_xlmr.py` | Fine-tunes XLM-RoBERTa in a plain PyTorch loop, so it runs the same on transformers 4.x and 5.x, CPU or GPU. |

## Task B

| File | What it does |
|------|--------------|
| `muril_b.py` | The six-class head over `muril.py`. Adds class-weighted loss and argmax decode. Task A's files are never read. |
| `tapt.py` | **The one thing that worked.** Masked-LM pass over in-domain text before any classifier exists, worth +2.9 points over stock MuRIL. Refuses Task A's CSVs by name: 319 of the 395 Task B test ids also sit in `binary_train.csv`. |
| `overnight_b.py` | Unattended sweep. Ranks ideas on a holdout, promotes winners to five folds, blends, decodes, and packages a zip per arm. Keeps a wall-clock budget so a session limit never truncates a fold, and rewrites `RESULTS.md` after every arm. |
| `decode_b.py` | Fits per-class decision weights on the OOF by coordinate ascent, because argmax maximizes accuracy and the task is scored on macro-F1. Reports argmax, in-sample and nested, and **refuses to write when nested loses** — which, so far, it always has. |
| `features.py` | Gazetteer / mood / address tags, behind `--tags`. Measured as noise; kept for the record. |
| `fetch_external.py` | Downloads the OffensEval-Dravidian corpora (CC BY 4.0). Kannada is tracked in `data/external/`; Tamil and Malayalam, used only by the grid's D1 cells, are fetched with `--text-only` and not tracked. |
| `sweep_b.sh` | The same arms as a shell ablation, one flag apart, for a local GPU box. |
| `fullfit_b.py` | Run 2: every idea trained on all rows with five seeds. No scores; ranks on the Run 1 holdout and reports agreement and class-rate drift. |
| `grid_b.py` | Run 3: factorial over TAPT corpus, vocabulary and auxiliary head on the fixed holdout. |
| `kaggle_*.ipynb` | One notebook per Kaggle run, indexed in [`NOTEBOOKS.md`](NOTEBOOKS.md). **`kaggle_d0v0_noaux_full.ipynb` is the current one.** |

## Usage

```sh
V=~/.venvs/hastika/bin/python

$V work/baseline_svm.py --task b --demojize          # floor, 0.5948

# the current recipe, D0_V0_noaux, on every comment and every row
$V work/tapt.py --corpus data/multiclass_train.csv data/external/offenseval_kn.csv \
    --val-frac 0 --min-words 1 --no-dedupe --out work/runs/tapt-d0v0-100   # ~25 min on a T4
$V work/muril_b.py --tag b_d0v0_noaux_full --model work/runs/tapt-d0v0-100 \
    --folds 1 --no-dedupe --aux-weight 0 --seeds 42 43 44 45 46             # ~95 min
$V work/make_submission.py --task b --pred work/runs/b_d0v0_noaux_full/predictions.csv

# the same recipe with a local score, as submitted in b_tapt_5f
$V work/tapt.py --out work/runs/tapt-muril
$V work/muril_b.py --tag b_tapt_5f --model work/runs/tapt-muril --folds 5
```

`--folds` picks what a run can tell you:

| `--folds` | each model trains on | scored on |
|---|---|---|
| `0` | 85% of rows | the fixed 472-row holdout |
| `5` (default) | 80% of rows, five models | out-of-fold predictions for every row |
| `1` | every row | nothing locally; CodaBench only |

Each run writes to `work/runs/<tag>/`: `predictions.csv`, `test_probs.npy`, and
`oof_probs.npy` for five-fold runs or `holdout_probs.npy` for `--folds 0`.
`ensemble.py` reads the OOF matrices and skips holdout runs automatically.

## Notes

- Text is **romanized** Kannada (Kanglish), not Kannada script — measured at 0%
  Kannada script across the corpus. That made a Latin-heavy encoder look
  promising, and it was measured: XLM-R scored 0.5561 against MuRIL's 0.5948 on
  the same holdout. MuRIL wins anyway.
- Every run prints **two** OOF scores. `best` picks each fold's checkpoint on the
  rows it then reports, so it is optimistic; `last` is unbiased. Compare arms on
  `last`.
- A 15% holdout ranks arms correctly but reads about two points high. Rank with
  it, never report it.
- MuRIL's MLM head emits a `bs x seq x 197285` logits tensor that cross-entropy
  upcasts to fp32. At `--bs 16 --max-len 192` that is a single 2.26 GiB
  allocation and it will not fit a T4. `tapt.py` defaults to `--bs 4
  --grad-accum 4`, same effective batch.

## The grid levers

Built for Run 3 and measured there. **None of them helped**: more TAPT text (D1) and the
auxiliary head landed inside the noise below `D0_V0_noaux`, and the extended vocabulary
cost 3 to 8 points in every cell. Full table in `TRAINING_RESULTS.md`. The code stays
because the flags are off by default and `corpora.py` is what `tapt.py` loads text with.

| File | What it does |
|------|--------------|
| `corpora.py` | One loader for every unlabelled-text source. CSVs, globs, `.txt`, `.jsonl`, or `@manifest`. Auto-detects the text column. Gates the three files carrying Task B test comments behind `--allow-transductive` and prints what it admitted, so a rules decision lands in the run log instead of being made silently in code. |
| `extend_vocab.py` | Adds the corpus's frequent word-forms to MuRIL's tokenizer, initializing each new embedding as the mean of the pieces it currently decomposes into. Feed the output to `tapt.py`. |
| `axes.py` | The act and target lexicons, with the measurements behind them. `act()` is weak supervision for the auxiliary head, not a classifier. |
| `split_b.py` | Materializes the fixed 472-row holdout as labelled CSVs under `data/holdout/`. |

Three levers, usable separately or together:

```sh
# 1. more adaptation data. Tamil + Malayalam measured as no gain in Run 3
$V work/tapt.py --extra 'data/external/*.csv' scraped/comments.txt
$V work/tapt.py --allow-transductive --extra data/binary_train.csv   # rules call

# 2. a tokenizer that fits this register. Measured as 3-8 points WORSE in Run 3
$V work/extend_vocab.py --min-freq 5 --out work/runs/muril-extended
$V work/tapt.py --model work/runs/muril-extended --out work/runs/tapt-extended

# 3. an auxiliary head on the act axis. Measured as no gain in Run 3
$V work/muril_b.py --tag b_aux --aux-weight 0.3
```

### Why these three were tried

Measured on this corpus, all within the one-point fold noise and therefore dead:
per-class decode weights (-0.009 nested), the OOV spelling resolver (+0.003),
explicit conjunction features (-0.003). Feature-level interventions do not move
this problem, because 72% of word types appear exactly once and character
n-grams already capture whatever lexical signal exists. Only the adaptation pass
has ever helped, at +2.9, and it works at the representation level. All three
levers above are representation-level for that reason.

Supporting numbers: 25.8% of validation tokens never appear in training; MuRIL
carries 197,258 subwords and this corpus emits 7,078 of them (3.6%) while
splitting each word into 2.17 pieces; 91% of comments contain no English
function word, so this is romanized Kannada rather than code-mixed text, which
is why XLM-R lost to MuRIL.
