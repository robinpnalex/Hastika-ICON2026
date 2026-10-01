# Released evaluation data

These four files were imported from the original repository
[`shankarb14/Hastika-ICON2026`](https://github.com/shankarb14/Hastika-ICON2026) after its
evaluation-data release. They are kept alongside, rather than replacing, the existing
unlabelled CodaBench input files used by the training code.

| file | rows | contents |
|---|---:|---|
| `hastika_binary_test.csv` | 806 | Task A test comments, without labels |
| `hastika_multiclass_test.csv` | 395 | Task B test comments, without labels |
| `hastika_binary_validation.csv` | 806 | Task A validation comments with `Label` |
| `hastika_multiclass_validation.csv` | 395 | Task B validation comments with `Hate Category` |

The labelled validation files have the same IDs as the existing
`binary_validation_inputs.csv` and `multiclass_validation_inputs.csv`. The existing
paths remain unchanged so the experiment notebooks and submission tools continue to
target the original CodaBench inputs.
