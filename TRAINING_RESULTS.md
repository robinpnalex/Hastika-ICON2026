# Training Results Log

This document tracks the experimental runs and their results for Task A (Binary Classification: Hate vs Non-Hate).

## Experiment 1: Baseline MuRIL
* **Model:** `google/muril-base-cased`
* **Preprocessing:** Default (HTML cleaning, mojibake repair)
* **Epochs:** 6
* **Best Epoch:** 6
* **Best Macro F1:** 0.7937
* **Output Directory:** `checkpoints/muril_task_a`

## Experiment 2: Demojized MuRIL
* **Model:** `google/muril-base-cased`
* **Preprocessing:** Default + Demojized (Emojis converted to English text descriptions)
* **Epochs:** 6
* **Best Epoch:** 5
* **Best Macro F1:** 0.8023
* **Output Directory:** `checkpoints/muril_task_a_demojized`
* **Notes:** Converting emojis to their text equivalents yielded an improvement of ~0.9% in the macro F1 score.

## Experiment 3: TF-IDF + LinearSVC Baseline
* **Model:** `TfidfVectorizer` (Word + Char n-grams) + `LinearSVC`
* **Preprocessing:** Default (HTML cleaning, mojibake repair)
* **5-Fold CV Macro F1:** 0.8039
* **Output Directory:** `work/runs/svm/`
* **Notes:** Surprisingly outperformed Baseline MuRIL, establishing a strong floor.

## Experiment 4: Demojized TF-IDF + LinearSVC Baseline
* **Model:** `TfidfVectorizer` (Word + Char n-grams) + `LinearSVC`
* **Preprocessing:** Default + Demojized (Emojis converted to English text descriptions)
* **5-Fold CV Macro F1:** 0.8103
* **Output Directory:** `work/runs/svm_demojize/`
* **Notes:** Best performing model so far, outperforming both standard TF-IDF and Demojized MuRIL.

## Next Planned Experiments
* **Experiment 5:** XLM-RoBERTa Base (`xlm-roberta-base`) with Demojized text.
