# Synthetic data for HASTIKA

## Files

| File | What |
|---|---|
| `batch1.jsonl` | 150 generated examples, schema-validated |
| `batch1_realised.jsonl` | batch1 + surface variation from `realism.py` |
| `realism.py` | injects elongation / caps / emoji / romanisation jitter at corpus-measured rates |
| `evaluate.py` | the two tests below — run this before training on any batch |

## Verdict: do not rely on this for Task A

| Batch | real-vs-synthetic AUC | OOF macro-F1 with augmentation |
|---|---|---|
| baseline (no synthetic) | — | 0.8032 |
| batch1 | 0.942 | 0.8053 (+0.0022) |
| batch1 + realism.py | 0.937 | 0.8063 (+0.0031) |

**AUC 0.94 means a plain TF-IDF linear model separates generated from real
comments almost perfectly.** The generated text is not drawn from the corpus
distribution, so a classifier trained on the mixture partly learns the generator's
fingerprint instead of the task.

The realism transform closed the *surface* gap (3+ char repeats 0% → ~23%, caps,
emoji, punctuation — all matched to measured corpus rates) and moved AUC by only
0.005. So the surface statistics were not the discriminator; the lexical
distribution is. Post-processing cannot fix that.

The +0.003 augmentation gain is inside CV noise, and at 150 rows (2.3% of the
training set) the test is underpowered regardless — it cannot resolve an effect
below ~0.005 either way.

## Methodology note

`evaluate.py` puts synthetic rows in **training folds only**. They never enter a
validation fold. Evaluating on a mixture measures how well a model fits your own
generations, which is not a quantity anyone wants.

## Where synthetic data is actually worth the effort

Task A is balanced (3286 / 3160) with 6,446 rows — the least promising place for
augmentation. Task B is the opposite:

    Gender 1362 | Political 559 | Others 449 | Religion 382 | Violence 221 | Geo-political 186

Macro-F1 weights all six equally, so Geo-political (186 rows) and Violence (221)
carry the same weight as Gender (1362) and are where the leaderboard is decided.
Generate for those two categories, not for Task A.

## Before using any batch

1. Run `evaluate.py`. Treat AUC > 0.85 as a stop sign.
2. Have a native Kannada speaker read a random sample of 30. Model-generated
   Romanised Kannada can be fluent and still read as non-native, and that is not
   something the generator can check about itself.
3. Scale to 1000+ rows before trusting the augmentation number at all.

## Label convention warning

The dataset's `Gender` category is assigned when a **gendered slur** (*sule*,
*boli*, *maga*) is present, regardless of the target's actual gender — e.g. real
rows abusing male cricketers are labelled `Gender`. Generating by the semantic
question "who is being attacked?" produces a different convention and injects
label noise. `batch1.jsonl` follows the dataset's convention, not the semantic one.
