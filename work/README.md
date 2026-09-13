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
| `fetch_external.py` | Downloads the OffensEval-Dravidian Kannada corpus (CC BY 4.0), the only extra labelled-free text Task B may use. |
| `sweep_b.sh` | The same arms as a shell ablation, one flag apart, for a local GPU box. |
| `kaggle_task_b.ipynb` | The sweep on Kaggle. Save Version → Save & Run All. |
| `kaggle_rebuild_winner.ipynb` | Only the winning arm, ~2 h instead of 8.5. |

## Usage

```sh
V=~/.venvs/hastika/bin/python

$V work/baseline_svm.py --task b --demojize          # floor, 0.5948
$V work/tapt.py --out work/runs/tapt-muril           # ~25 min on a T4
$V work/muril_b.py --tag b_tapt_5f --model work/runs/tapt-muril --folds 5
$V work/make_submission.py --task b --pred work/runs/b_tapt_5f/predictions.csv
```

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

## Added for the next round

| File | What it does |
|------|--------------|
| `corpora.py` | One loader for every unlabelled-text source. CSVs, globs, `.txt`, `.jsonl`, or `@manifest`. Auto-detects the text column. Gates the three files carrying Task B test comments behind `--allow-transductive` and prints what it admitted, so a rules decision lands in the run log instead of being made silently in code. |
| `extend_vocab.py` | Adds the corpus's frequent word-forms to MuRIL's tokenizer, initializing each new embedding as the mean of the pieces it currently decomposes into. Feed the output to `tapt.py`. |
| `axes.py` | The act and target lexicons, with the measurements behind them. `act()` is weak supervision for the auxiliary head, not a classifier. |
| `split_b.py` | Materializes the fixed 472-row holdout as labelled CSVs under `data/holdout/`. |

Three levers, usable separately or together:

```sh
# 1. more adaptation data -- the only intervention ever measured to help
$V work/tapt.py --extra 'data/external/*.csv' scraped/comments.txt
$V work/tapt.py --allow-transductive --extra data/binary_train.csv   # rules call

# 2. a tokenizer that fits this register
$V work/extend_vocab.py --min-freq 5 --out work/runs/muril-extended
$V work/tapt.py --model work/runs/muril-extended --out work/runs/tapt-extended

# 3. an auxiliary head on the act axis
$V work/muril_b.py --tag b_aux --aux-weight 0.3
```

### Why these three and not others

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
