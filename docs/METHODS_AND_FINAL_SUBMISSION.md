# Methods used in the HASTIKA experiments

This document is a consolidated methods record for the paper. It describes the
approaches that were tried across Task A and Task B, the evaluation protocol used to
compare them, and the exact systems whose predictions were submitted.

The experiment ledger remains the authoritative source for individual run details and
logs: [`EXPERIMENTS.md`](EXPERIMENTS.md). Notebook inventories are available in the
Task A and Task B indexes: [`task_a/NOTEBOOKS.md`](task_a/NOTEBOOKS.md) and
[`task_b/NOTEBOOKS.md`](task_b/NOTEBOOKS.md).

## 1. Tasks and data

HASTIKA contains short Kannada-English code-mixed YouTube comments written largely in
Romanized Kannada (Kanglish).

| task | problem | labels | labelled rows used for final fitting |
|---|---|---|---:|
| Task A | binary hate-speech detection | `Hate`, `Non-Hate` | 7,193 |
| Task B | hate-category classification | `Gender`, `Geo-political`, `Others`, `Political`, `Religion`, `Violence` | 3,532 |

The labelled data available for the final fit consisted of the original training data
plus the released labelled validation data. For honest model comparison, we created a
fixed 85/15 split before training the candidate models:

- Task A: 6,114 training-side rows and a 1,079-row holdout.
- Task B: 3,002 training-side rows and a 530-row holdout.

The holdouts were used to compare recipes, not to train the final submitted models.
Once a recipe was selected, the final fitting notebooks trained on all available
labelled rows. The released test labels were unavailable, so test predictions could
not be scored locally.

Unless a run is explicitly marked as transductive, test text and test-derived labels
were excluded from training, TAPT, and model selection. Cross-task label derivation and
test-set pseudo-labelling were investigated as possibilities but were not used in the
recorded final submission.

## 2. Evaluation and reproducibility protocol

The primary metric was macro-F1 because the tasks contain minority categories and a
model that predicts the majority class can otherwise look deceptively strong. Accuracy
was reported as a secondary metric. Task B also reported per-class F1 because
`Geo-political` and `Violence` were substantially harder than the larger categories.

The evaluation protocol had three levels:

1. **Fixed holdout:** a single split was kept unchanged across comparable arms. This
   made paired bootstrap comparisons possible.
2. **Five-fold out-of-fold (OOF) evaluation:** used for the earlier MuRIL/SVM blends and
   for methods where a single holdout could be noisy. Each row received a prediction
   from a model that did not train on that row.
3. **Full-data fitting:** after a recipe was chosen, models were retrained on all
   labelled rows and used to predict the released test data.

Random seeds were used to measure training variance rather than to select a lucky
model. The same split and probability-column order were preserved when combining
models. Probability ensembles were averaged before taking the final argmax label.

The experiments used automatic checks for row IDs, prediction shapes, missing outputs,
and degenerate models. Gemma runs also used a collapse guard: if more than 90--95% of a
training sample's predictions became one class, the run was rejected or retried with a
new seed instead of silently entering an ensemble.

## 3. Text preprocessing

### 3.1 De-emojization

For MuRIL/BERT-style wordpiece models, emoji removal was enabled. MuRIL's wordpiece
tokenizer has limited emoji coverage and can map a whole whitespace-delimited word to
`[UNK]` when an emoji is attached. De-emojization reduced the observed unknown-token
rate substantially on the training corpus. SentencePiece-based decoder models, such as
Gemma, were not dependent on this preprocessing and used their native tokenizer.

URLs and other routine text normalization were handled by the shared preprocessing
utilities. The code preserved the original comment IDs so predictions could be joined
back to the required submission format.

### 3.2 Transliteration and normalization

Several normalization ideas were tested, including character repetition normalization
and a light transliteration-style spelling normalization. They did not materially
improve the character baseline. The data's large vocabulary was found to be driven more
by agglutinative morphology and inflection than by a simple spelling-noise problem.
Consequently, aggressive transliteration was not used in the final system.

## 4. Classical and MuRIL methods

### 4.1 TF-IDF and LinearSVC baseline

The first strong baseline used TF-IDF features with a linear support-vector classifier.
Word and character n-grams were combined, and class balancing was used where
appropriate. A calibrated LinearSVC produced class probabilities for evaluation and
blending.

The character features were especially useful because code-mixed Romanized Kannada has
many rare surface forms. Character n-grams can share information across related spellings
and morphological variants that a word vocabulary treats as unrelated.

### 4.2 MuRIL fine-tuning

MuRIL was selected because its pretraining includes transliterated Indic text and is
therefore better matched to Kanglish than a generic multilingual encoder. The main
MuRIL recipe used:

- demojized input;
- mean-plus-max token pooling;
- reinitialization of the top encoder layers in the transfer-learning experiments;
- layer-wise learning-rate decay to protect lower pretrained layers;
- FGM adversarial perturbation of embeddings;
- EMA of model weights;
- label smoothing;
- multiple random seeds and averaged predictions.

The classification head was trained on top of the encoder representation. The top-layer
reinitialization experiments compared reinitializing two layers, one layer, and no
layers. The motivation was that the final pretrained layers are highly specialized for
masked-language modelling and can transfer poorly to a small supervised classification
set. The experiments showed that the best number of reinitialized layers was task
dependent: one layer was favoured in the strongest Task B MuRIL recipe, while two layers
were competitive in parts of the Task A funnel.

### 4.3 MuRIL plus TF-IDF ensembles

The MuRIL and TF-IDF/SVM models made complementary errors, so their probability outputs
were blended. The blend weight and, for Task A, the decision threshold were fitted only
from aligned OOF predictions. The original Task A blend used approximately 57% SVM and
43% MuRIL. A later one-layer reinitialization comparison refitted the blend instead of
reusing the old weight.

The OOF results were stronger than one individual component, but the first external
validation submission underperformed its OOF estimate. This was an important reason to
keep OOF results and externally scored results labelled separately.

### 4.4 MuRIL regularization and ablations

The following methods were tested during the MuRIL stage:

- **FGM:** adds a small adversarial perturbation to embedding parameters during a second
  forward/backward pass. It encourages a locally smoother decision function and was
  useful for rare code-mixed surface forms.
- **EMA:** maintains an exponential moving average of weights and evaluates the averaged
  parameters, reducing sensitivity to individual minibatch updates.
- **R-Drop:** performs two stochastic dropout forward passes for the same example and
  adds a symmetric KL-divergence penalty between their class distributions. It was
  useful as an ensemble member in some Task B runs, but was not uniformly better as a
  replacement model.
- **Frozen-embedding SVM:** an RBF SVM was trained on frozen MuRIL embeddings. It did
  not recover the performance lost by freezing the encoder and was rejected.
- **MuRIL-large:** a larger encoder was tested with memory-conscious settings. It was
  not adopted for the final submission.

## 5. Task-adaptive pretraining (TAPT)

TAPT adapts a pretrained language model to the target-domain text before supervised
classification. It uses comments without reading their labels during the language-model
stage, then initializes the classifier from the adapted encoder.

### 5.1 MuRIL TAPT

For MuRIL, TAPT used masked-language-model training on permitted in-domain and related
Kannada text, followed by the same supervised MuRIL recipe as the control. The OOF Task A
experiment improved the MuRIL component from approximately 0.7894 to 0.8128 macro-F1,
although the exact magnitude is subject to the corpus and evaluation-side text rules.
Task B also showed a positive TAPT signal in the earlier MuRIL line.

### 5.2 Gemma TAPT

For Gemma, TAPT used LoRA next-token training on the permitted comment corpus. The TAPT
adapter was then loaded into the supervised classifier, which trained a classification
head and continued LoRA fine-tuning. Later Task A and Task B bag experiments found that
independent TAPT runs could be averaged to reduce seed variance. These later TAPT-bag
systems are documented as research candidates; they were not the artifacts used in the
recorded final submission.

## 6. Decoder LLM method: Gemma-4-12B

Gemma became the strongest model family for both tasks. The implementation used Gemma
as a decoder representation model, not as a generative label decoder.

### 6.1 Quantization and parameter-efficient fine-tuning

The model was loaded in 4-bit NF4 quantization with double quantization and fp16
computation on Kaggle T4 GPUs. QLoRA adapters were trained instead of updating all
Gemma weights. The default LoRA configuration was:

- rank `r = 16`;
- attention projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`;
- MLP projections: `gate_proj`, `up_proj`, `down_proj`;
- LoRA alpha `2r`;
- LoRA dropout `0.05`;
- gradient checkpointing;
- fp16 mixed precision.

The large language-model head was not loaded. This reduced memory use and avoided
projecting every token through Gemma's very large vocabulary when only classification
was required.

### 6.2 Classification head and prompt

Each comment was placed into a short task-specific prompt. The model's hidden state at
the final prompt token was extracted and passed through a dropout layer followed by a
new linear classification head:

```text
Gemma backbone -> last-token hidden state -> dropout -> linear classification head
```

The head was initialized from scratch. Dropout was applied immediately before the head
to regularize the newly initialized task-specific mapping. The prompt ended with an
answer cue such as “Category:” or “Hate or Non-Hate?:”, so the final hidden state
represented the model's state immediately before producing the answer.

The supervised loss used balanced class weights and label smoothing of 0.05. The
standard learning rates were `1e-4` for LoRA parameters and `5e-4` for the new head,
with AdamW, weight decay, warmup, and a cosine schedule. The effective batch size was
16 through gradient accumulation.

### 6.3 Gemma ablations

The Gemma experiments tested the following decisions:

- three versus four classifier epochs;
- standard short prompt versus prompts containing organiser-provided category
  definitions;
- ordinary balanced class weights versus square-root inverse-frequency weights;
- R-Drop;
- frozen-backbone/head-only training;
- TAPT before classification;
- LoRA capacity and target-module choices;
- averaging models from multiple seeds and independent TAPT runs;
- a possible second model family, Qwen3-8B, as an ensemble member.

The Task A holdout showed that three classifier epochs were safer than four. The Task B
final comparison favoured four epochs. The definitions prompt and square-root weighting
did not provide a reliable improvement. R-Drop was inconclusive as a standalone recipe
but produced useful diversity in later ensembles. TAPT and independent model averaging
were the most promising later directions.

## 7. Ensembles

Two kinds of ensemble were used.

### 7.1 Heterogeneous probability blending

During the MuRIL stage, TF-IDF/SVM and MuRIL probabilities were combined with a weight
selected from OOF predictions. This helped because the two representations captured
different signals: character morphology and pretrained multilingual semantics.

### 7.2 Same-family seed averaging

For Gemma, probability averaging across independently trained seeds reduced the effect
of unstable optimization. A model was included only if its row IDs and probability
shape matched and it did not collapse to one class. The final Gemma submissions used
fixed equal averaging rather than selecting a seed after seeing test predictions.

Later TAPT-bag experiments extended this idea by averaging models from independent TAPT
runs. Their holdout results were encouraging, but they happened after the recorded
competition submission and must be described as follow-up research rather than as the
submitted system.

## 8. Methods tried but not used in the final submission

The following approaches were evaluated or prepared but were not part of the recorded
final artifacts:

- task-domain TAPT variants that failed or were superseded by safer implementations;
- R-Drop as a replacement recipe where the paired comparison was inconclusive;
- context/topic/mood/address tags;
- external-label mixing and cross-task-derived labels;
- English translation followed by an English hate-speech model;
- frozen MuRIL embeddings with an RBF SVM;
- MuRIL-large;
- larger or alternative Gemma LoRA configurations;
- definitions-prompt and square-root-class-weight variants;
- later Gemma TAPT, TAPT+R-Drop, TAPT-bag, and Qwen experiments.

The cross-task/test-derived-label line is also excluded on methodological and
transductive grounds. It should not be presented as part of the clean final method.

### 8.1 Complete idea-status inventory

The following inventory makes the coverage explicit. “Measured” means that a controlled
run produced a score; “prepared” means that a notebook or implementation exists but it
was not used to select the submitted system; and “not built” means the idea remained a
proposal.

| idea | task(s) | status and paper treatment |
|---|---|---|
| MuRIL TAPT | A, B | **Measured**; positive representation-adaptation signal, especially in the earlier MuRIL OOF experiments |
| External labelled corpus | A | **Measured** on the TF-IDF floor; rules-sensitive and not used in the final submission |
| One versus two versus zero reinitialized layers | A, B | **Measured**; the best count was encoder/task dependent |
| Five-seed averaging | A, B | **Measured**; reduces optimization variance but increases compute |
| OOF threshold tuning | A | **Measured** on the SVM; later Gemma final used direct argmax rather than a tuned test threshold |
| OOF blend-weight refitting | A | **Measured**; required when the MuRIL component changed |
| R-Drop | A, B | **Measured**; often useful as a diverse ensemble member, not consistently superior alone |
| Ten classifier epochs | A, B | **Screened/prepared** in the MuRIL funnel; later Gemma epoch comparisons selected 3 epochs for A and 4 for B |
| MuRIL-large | A, B | **Measured/screened**, not selected for final submission |
| Whole-word masking for TAPT | A, B | **Proposed, not completed** |
| Self-distillation with OOF soft labels | A, B | **Proposed, not built** |
| Independent TAPT encoders | A, B | **Measured later** through TAPT bags; post-submission research, not the recorded final artifacts |
| Context-conditional decoding | B | **Prepared/proposed**; a TF-IDF nested signal existed, but it was not adopted for the submitted Gemma system |
| XLM-R as a third ensemble member | A | **Prepared/downgraded** after weak diversity evidence; not used |
| Pseudo-labelling released inputs | A, B | **Proposed but excluded** from the clean final method |
| Supervised contrastive auxiliary loss | A, B | **Proposed, not built** |
| Frozen MuRIL embeddings plus RBF SVM | A | **Measured and rejected** |
| Vocabulary extension during TAPT | B | **Measured and rejected**; it reduced performance |
| Auxiliary act/category head | B | **Measured and rejected** |
| Focal loss | B | **Measured and rejected** |
| Topic, mood, address, gazetteer and NER-style features | A, B | **Measured and rejected or inconclusive** |
| Extra Tamil/Malayalam TAPT text | B | **Measured and rejected** |
| Synthetic generated data | A | **Measured; gain inside noise**, not used |
| Character normalization and transliteration-lite | A, B | **Measured; neutral or negative**, not used |
| Stopword removal and stemming | A, B | **Checked and rejected** because they remove useful code-mixed or morphological signal |
| Profanity/insult-density features | A | **Measured and rejected**; most errors contained no explicit profanity |
| Character-only SVM | A | **Measured**; stronger as a standalone baseline, but not proven as a useful third ensemble member |
| Cross-task-derived labels | A | **Measured as a transductive diagnostic**, excluded from the final method |
| Translation to English plus hateBERT | A | **Measured and rejected**; translation damaged abusive-language cues |
| LLM screening and Sarvam-1 | B | **Measured**; Gemma-4-12B was selected, Sarvam-1 was rejected |
| Gemma plus SVM or MuRIL | A, B | **Measured**; improvements were absent or inside noise, so Gemma was used alone |
| Gemma prompt definitions | A, B | **Measured/prepared**; no reliable replacement for the short prompt |
| Gemma learning-rate change | A, B | **Prepared; not selected** |
| Gemma LoRA rank and target-module capacity | B | **Prepared; not selected** |
| Frozen Gemma backbone with head-only training | B | **Prepared ablation; not selected** |
| Gemma TAPT and TAPT + R-Drop | A, B | **Measured after the recorded submission**; strong follow-up results, not the submitted artifacts |
| Qwen3-8B second-family ensemble | A | **Prepared/pending**; not part of the recorded submission |

This table is intentionally broader than the final-method narrative: a paper should use
it to decide what belongs in the main ablation table, what belongs in an appendix, and
what should be mentioned only as a rejected or unrun idea.

## 9. Exact final submission systems

The repository records the actual prediction files used for submission separately from
later research candidates. The final systems were both full-data Gemma fits.

### 9.1 Task A — Run 28

- **Model:** Gemma-4-12B with 4-bit QLoRA.
- **Input:** short binary hate-speech prompt and the original comment text.
- **Supervised training:** three epochs; balanced classification loss with label
  smoothing; LoRA fine-tuning plus a randomly initialized dropout classification head.
- **Seeds:** 42 and 43, trained independently on all 7,193 labelled rows for the final
  fit.
- **Selection evidence:** seed 42 scored 0.8563 macro-F1 on the fixed 1,079-row holdout;
  the two-seed recipe measured 0.8610 in the matched epoch comparison. The char-SVM
  baseline scored 0.8117, and adding it to Gemma did not help.
- **Submission artifact:**
  [`submissions/task_a/final/Project-MANAS_taskA.csv`](../submissions/task_a/final/Project-MANAS_taskA.csv)
  and the retained source ZIP `RECOMMENDED_a28_gemma.zip`.

### 9.2 Task B — Run 23

- **Model:** Gemma-4-12B with the same 4-bit QLoRA and last-token classification-head
  design.
- **Input:** short six-category prompt and the original comment text.
- **Supervised training:** four epochs; balanced class weights; label smoothing;
  LoRA fine-tuning and the new classification head.
- **Seeds:** 42--45, trained independently on all 3,532 labelled rows for the final
  fit.
- **Selection evidence:** the four-model recipe scored 0.6872 macro-F1 on the fixed
  530-row holdout, improving on the earlier MuRIL result of 0.6151 and the two-seed
  Gemma result of 0.6792.
- **Submission artifact:**
  [`submissions/task_b/final/Project-MANAS_taskB.csv`](../submissions/task_b/final/Project-MANAS_taskB.csv).
  The source ZIP was generated in Kaggle but is not retained in this checkout.

### 9.3 What was not used in the recorded submission

The later TAPT results were stronger on the fixed holdouts: Task A TAPT ensembles
reached 0.8702--0.8721 and Task B TAPT pairs reached approximately 0.7026--0.7056.
Those runs are important follow-up findings, but they were produced after the recorded
Run 28/Run 23 submission artifacts and should be labelled as post-submission research
in the paper.

## 10. Limitations and reporting cautions

- The released test labels were unavailable, so test performance cannot be reported
  directly from this repository.
- Some early comparisons used external validation scores, while later comparisons used
  fixed local holdouts or OOF predictions. These numbers must not be presented as if
  they came from one common evaluation split.
- Small Task B minority classes caused noticeable seed variance. A single seed can be
  misleading; averaged predictions and paired bootstrap intervals are more informative.
- Gemma occasionally entered a collapsed training state. The collapse guard and retry
  logic are part of the practical method and should be described for reproducibility.
- The final submitted systems did not use test-derived labels, pseudo-labelling, or
  cross-task label overrides.

## 11. Suggested paper method structure

The eventual paper can turn this record into the following sections:

1. Dataset and task formulation.
2. Kanglish preprocessing and exploratory observations.
3. Classical TF-IDF/SVM and MuRIL baselines.
4. Encoder adaptation and regularization: reinitialization, FGM, EMA, and R-Drop.
5. TAPT and parameter-efficient Gemma classification.
6. Evaluation protocol, seed handling, and probability averaging.
7. Ablation results and error/variance analysis.
8. Final submission systems and limitations.
