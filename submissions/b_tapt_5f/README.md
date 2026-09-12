# b_tapt_5f

The Task B submission. Domain-adapted MuRIL, five folds, six epochs.

| | |
|---|---|
| unbiased OOF macro-F1 | **0.6013** |
| selected (`best`) OOF macro-F1 | 0.6023 |
| measured on | 3,143 training rows, 5-fold out-of-fold |
| encoder | `google/muril-base-cased` after `work/tapt.py` |
| seeds | 42 (split seed 42) |
| produced by | `work/overnight_b.py`, 2026-09-12, 8.44 h on a T4 |

## Why this one

It was the only five-fold run in the sweep to beat the TF-IDF floor of 0.5948.
Stock MuRIL over the same five folds scored 0.5727, so the masked-LM adaptation
pass is worth about +2.9 points on its own. Nothing else helped: XLM-R scored
0.5561, HingRoBERTa 0.5678, focal loss and R-Drop both landed under stock MuRIL,
and mDeBERTa collapsed to 0.1002 because the shared learning rate is too high
for it.

Blending did not pay either. `work/ensemble.py` weight-searched over every
five-fold run plus the SVM and its nested estimate came out at 0.6002, below
this single model.

## Predicted distribution

The floor abandoned `Violence`, calling it on 6 of 395 test rows against a 7.0%
training rate. Under macro-F1 that class is worth a sixth of the score whatever
its support, so that alone cost several points. This run tracks the prior:

| class | training | this run | TF-IDF floor |
|---|---|---|---|
| Gender | 43.1% | 44.8% | 53.2% |
| Political | 17.7% | 17.0% | 18.0% |
| Others | 14.2% | 14.2% | 11.7% |
| Religion | 12.2% | 9.1% | 11.1% |
| Geo-political | 5.9% | 8.9% | 4.6% |
| Violence | 7.0% | 6.1% | 1.5% |

## Reproduce

```bash
python -u work/tapt.py --out work/runs/tapt-muril
python -u work/muril_b.py --tag b_tapt_5f --model work/runs/tapt-muril \
    --folds 5 --seeds 42 --epochs 6
python work/make_submission.py --task b \
    --pred work/runs/b_tapt_5f/predictions.csv --out b_tapt_5f.zip
```

Roughly 25 minutes for the adaptation pass and 90 minutes for the five folds on
a T4. `work/kaggle_rebuild_winner.ipynb` is these three commands as a notebook.
