# Submission Format (quick reference)

Submit a single `predictions.csv`, zipped, to the matching CodaBench task.

## Task A — Binary
Header: `id,label`
Labels: Hate | Non-Hate  (1 | 0 also accepted)

## Task B — Fine-Grained
Header: `id,label`
Labels: Gender | Political | Religion | Geo-political | Violence | Others
(GEN | POL | REL | GEO | VIO | OTH also accepted)

Rules: one row per id from the input file; UTF-8; keep ids unchanged; zip only predictions.csv.

## Building a ZIP from an already-finished run

A run that saved only `test_probs.npy` still has everything a submission needs. Both
`muril.py` and `baseline_svm.py` write the probability matrix in a fixed column order —
`[Non-Hate, Hate]` for Task A, the six `LABELS` in order for Task B — so labels can be
recovered afterwards without retraining.

```sh
# from a predictions.csv the run already wrote
python -m hastika.common.submission --task a \
    --pred artifacts/runs/<tag>/predictions.csv --out <tag>.zip

# from a probability matrix, when only that was saved
python -m hastika.common.submission --task a \
    --probs artifacts/runs/<tag>/test_probs.npy --out <tag>.zip

# the same, applying a threshold fitted elsewhere instead of 0.5
python -m hastika.common.submission --task a \
    --probs artifacts/runs/<tag>/test_probs.npy --threshold 0.46 --out <tag>.zip
```

`--threshold` is Task A only and applies to `p(Hate)`. It exists so a threshold fitted on
out-of-fold probabilities can be applied to an old run without retraining anything.

Either form runs the same checks before writing: header `id,label`, one row per released
id, no duplicates, only allowed labels, and a flat archive containing one bare
`predictions.csv`. If any check fails nothing is written.
