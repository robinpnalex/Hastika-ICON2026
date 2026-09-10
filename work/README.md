# HASTIKA Task A — advanced experiments

This directory contains the optional five-fold CV, SVM, transformer, and ensemble
pipeline. The simpler supported workflow is documented in the repository root
README and `docs/TASK_A_GUIDE.md`. Both workflows use the root `uv` environment.

| File | What it does |
|------|--------------|
| `prep.py` | Text cleaning. The released CSVs are **mojibake** (UTF-8 bytes decoded as latin-1), so Kannada script and emoji arrive mangled. Undoes that, unescapes HTML entities, strips `<br>`, normalizes URLs/@mentions. |
| `baseline_svm.py` | TF-IDF char(2-5) + word(1-2) n-grams → LinearSVC. 5-fold CV. Seconds to run, no GPU. The floor XLM-R must beat. |
| `muril.py` | **Primary model.** Dedicated MuRIL fine-tune: meanmax pooling, layer-wise LR decay, top-layer re-init, FGM adversarial training, EMA, multi-seed, OOF threshold tuning. Defaults are the tuned recipe. |
| `train_xlmr.py` | Fine-tunes XLM-RoBERTa. Plain PyTorch loop (no `Trainer`), so it runs the same on transformers 4.x and 5.x, CPU or GPU. |
| `ensemble.py` | Blends any runs that have OOF matrices, by Dirichlet weight search seeded with the single-model corners. |
| `scripts/make_submission.py` | Validates a `predictions.csv` against the task rules and writes a flat zip. |

## Usage

```bash
uv run --extra cu128 python work/baseline_svm.py
uv run --extra cu128 python work/train_xlmr.py --tag xlmr-holdout
uv run --extra cu128 python work/train_xlmr.py --folds 5 --tag xlmr-cv
uv run --extra cu128 python scripts/make_submission.py \
  --pred work/runs/xlmr-cv/predictions.csv --task a
```

Each run writes to `work/runs/<tag>/`: `predictions.csv`, `test_probs.npy`
(softmax probabilities on the validation inputs), and OOF/holdout probabilities
for threshold tuning or ensembling.

## Notes

- Text is **romanized** Kannada (Kanglish), not Kannada script — XLM-R pretrained
  on Kannada script, so its edge over char n-grams is smaller than usual. Worth
  comparing against `google/muril-base-cased`, which saw transliterated Indic text.
- Token lengths: p50 18, p95 68, p99 121 → `--max-len 128` truncates ~1% of rows.
- Batches pad to the longest item in the batch, not to `--max-len`; on CPU that is
  roughly a 2x speedup.
