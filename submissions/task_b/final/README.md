# Task B final submission

This is the Task B prediction file actually used for the final submission on
2026-10-07.

| file | what it is |
|---|---|
| `Project-MANAS_taskB.csv` | the submitted predictions: `id,label`, 396 rows, one per id of `data/raw/hastika_multiclass_test.csv` |

**Produced by** Task B Run 23, `notebooks/task_b/23_gemma_epochs_seeds_final.ipynb`.
The run trained four Gemma-4-12B 4-bit QLoRA classifiers (seeds 42--45) on all 3,532
labelled rows. Its fixed 530-row holdout result was **0.6872 macro-F1**.

The notebook generated `RECOMMENDED_b23_gemma.zip` in Kaggle. The final artifact retained
in this checkout and used for submission is the CSV above; the ZIP is not currently
stored under this repository's final folder.

Predicted class counts: Gender 171, Geo-political 29, Others 49, Political 72,
Religion 43, Violence 32.
