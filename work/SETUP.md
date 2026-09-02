# Running the sweep on the GPU machine

Copy the whole repo (or just `data/` + `work/`) across, then:

```sh
python -m venv .venv && . .venv/bin/activate

# torch first, matched to that machine's CUDA. Check with nvidia-smi.
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r work/requirements.txt

python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```

Then:

```sh
PY=python bash work/run_all.sh          # full sweep + blend + zipped submission
LARGE=0 bash work/run_all.sh            # base models only, ~1.5h
FOLDS=0 EPOCHS=6 bash work/run_all.sh   # quick holdout pass to sanity-check the box
```

Output lands in `work/runs/<tag>/` and the final upload is
`work/runs/ensemble/predictions.zip`.

## Knobs that matter

| Flag | Why |
|---|---|
| `--demojize` | Required for MuRIL. Its wordpiece vocab has no emoji, and BERT wordpiece drops the *entire* whitespace word containing one to `[UNK]` — measured 0.80% → 0.05% UNK on the training set. Roughly neutral for sentencepiece models, which keep emoji natively. |
| `--epochs 6` | XLM-R needed 2 full epochs on this data just to escape the trivial solution (loss pinned at ln 2 ≈ 0.693). Three is not enough. |
| `--amp auto` | bf16 on Ampere+, fp16 with loss scaling on older cards, off on CPU. |
| `--folds 5` | Required for `ensemble.py` — a holdout run produces no OOF matrix to fit blend weights on. |
| `--seed 42` | Do not change per-model. All OOF matrices must share CV splits or the blend weights are fitted on misaligned rows. `baseline_svm.py` hardcodes the same seed. |

## Expected numbers

- TF-IDF char n-gram + LinearSVC floor: **0.803** OOF macro-F1 (measured, 8 seconds)
- XLM-R, 3 epochs, CPU: 0.566 → 0.716 → (epoch 3) — under the floor, too few epochs
- MuRIL is the best single prior: its pretraining included transliterated Indic,
  which is what Kanglish is. Expect roughly +2–5 over XLM-R based on published
  DravidianLangTech code-mixed results.
- The blend should beat every individual member; `ensemble.py` seeds the search
  with the single-model corners so it cannot come out worse.

## Measured: why transliteration is NOT worth building

The usual argument for back-transliterating romanized Kannada into Kannada script
is that it normalizes spelling variants. On this dataset that argument does not
survive measurement.

Vocabulary is badly fragmented — 21,363 types from 71,159 tokens, 70.4% hapax
(English at this size is ~45%). But the cause is **agglutinative morphology**,
not spelling noise: 65.2% of hapax types share a 5-char stem with another type,
and the clusters are verb paradigms (`matha*` 106 forms, `madid*` 86, `matad*` 74
— inflections of *maadu* "to do" and *maataadu* "to speak").

Script conversion does not merge morphological variants: ಮಾಡ್ತಾ and ಮಾಡ್ತಾರೆ stay
distinct. Two experiments, 5-fold CV, same splits:

| Preprocessing | Types | Hapax | macro-F1 |
|---|---|---|---|
| raw | 21,363 | 70.4% | 0.8032 |
| char normalization (3+ repeats → 2) | 21,228 | 70.2% | 0.8003 |
| char normalization, aggressive | 20,309 | 69.4% | 0.8017 |
| "transliteration-lite" (th→t, dh→d, … + vowel collapse) | 19,097 | 68.4% | 0.8043 |

Transliteration-lite simulates exactly the merging a real transliterator performs.
It removes 10.6% of the vocabulary and buys **+0.001 macro-F1** — inside CV noise.

Conclusion: the normalization pathway is a dead end here. Subword tokenization
already handles agglutination (it decomposes `madthare` into stem + suffixes),
which is why the char n-gram SVM is at 0.803. Prefer MuRIL, whose advantage comes
from a different mechanism — pretraining on transliterated Indic gives it
representational alignment with no cascading error. Keep transliteration only as
late-stage ensemble diversity, if at all.

## The MuRIL recipe (`work/muril.py`)

This is the primary model. Every non-default choice is tied to something measured
on this dataset rather than copied from a generic recipe.

| Choice | Default | Why, for *this* data |
|---|---|---|
| `--demojize` | **on** (off in `train_xlmr.py`) | BERT wordpiece has no emoji and maps the whole surrounding whitespace word to `[UNK]`. Measured 0.80% → 0.05% UNK. Sentencepiece models don't need it. |
| `--pooling meanmax` | mean ⊕ max over tokens | Comments are short (p50 18 tokens) and the signal is often one or two slur tokens. Max-pooling catches those; `[CLS]` washes them out. |
| `--reinit-layers 2` | re-init top 2 encoder layers | Top layers are the most MLM-specialized and transfer worst on small data (Zhang et al., *Revisiting Few-sample BERT Fine-tuning*). |
| `--llrd 0.9` | layer-wise LR decay | MuRIL's value is its pretrained handling of transliterated Indic — decay protects lower layers from being overwritten by 6.4k rows. |
| `--fgm` | on, ε=1.0 | 70% of vocabulary types are hapax, so their embeddings overfit freely. Adversarial perturbation forces local flatness. ~1.8× step cost. |
| `--ema` | on, decay 0.999 | Fine-tuning at this size is high-variance; EMA weights are evaluated instead of raw. |
| `--epochs 6`, `--warmup 0.06` | | XLM-R needed 2 full epochs just to escape the trivial solution here. 3 is not enough. |
| `--evals-per-epoch 2` | | Val F1 moves sharply between epochs (0.566 → 0.716); epoch-level checkpoint selection is too coarse. |
| `--seeds 42 1337` | average 2 fine-tunes | The most reliable single gain at this dataset size. |
| label smoothing 0.05 | | Annotation noise is expected in crowd-labelled social media. |
| no class weighting | | Labels are near-balanced: 3286 Non-Hate / 3160 Hate. |

Threshold tuning on OOF is restricted to [0.30, 0.70] — with balanced classes the
honest optimum is near 0.5, and an extreme threshold means a degenerate model.
It is only applied if it beats 0.5 by more than 0.002.

Smoke-test the recipe on a new machine before committing to a full run:

```sh
python work/muril.py --tag smoke --limit 160 --folds 2 --epochs 1 \
    --bs 8 --max-len 32 --threads 4
rm -rf work/runs/smoke        # else ensemble.py will auto-discover it
```

### Cost

FGM roughly doubles step cost and `--seeds 42 1337` doubles it again, so the
default MuRIL run is ~4× a plain fine-tune: budget roughly 1.5–2.5h for 5 folds ×
2 seeds × 6 epochs on an Ampere card. Drop `--seeds` to one value or pass
`--no-fgm` to cut that. `--rdrop 0.5` adds another ~1.4× and is off by default.
