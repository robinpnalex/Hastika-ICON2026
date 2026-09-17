"""Context-conditional prior correction, applied to probabilities after training.

THE OBSERVATION
---------------
A violent action word does not mean Violence. What it means depends on what else
is in the comment. Measured on multiclass_train (3,143 deduplicated rows), among
the 420 comments containing a violent action word:

    also contains     n    Gender  Geo   Others  Polit  Relig  Violence
    political        71     0.03   0.08   0.00   0.56   0.17    0.15
    religion         76     0.03   0.07   0.01   0.16   0.61    0.13
    geo              50     0.04   0.26   0.00   0.16   0.22    0.32
    media            78     0.31   0.06   0.27   0.08   0.10    0.18
    nothing         216     0.40   0.03   0.10   0.07   0.06    0.34
    corpus prior   3143     0.43   0.06   0.14   0.18   0.12    0.07

So "hodibeku" next to BTV is Others at 0.27 and Violence at 0.18; the same word
with no competing target is Violence at 0.29 and Others at 0.06. Violence wins
only when no target group is named, and even then it trails Gender.

WHY THIS AND NOT `decode.py`
----------------------------
`decode.py` fits ONE weight per class and has lost its nested check on every run
ever tried. It cannot express the table above, because the right correction for
Violence flips sign depending on context. This fits a lift per CELL instead.

Measured on the calibrated TF-IDF SVM, 5-fold, weight chosen by an inner split of
each training fold so the rows being scored never influence it:

    no correction   0.5630   (Violence F1 0.24)
    nested          0.5823   (Violence F1 ~0.35 at the selected weight)

TWO THINGS TO CHECK BEFORE BELIEVING IT
---------------------------------------
1. That base is weak. Plain LinearSVC argmax scores 0.5948 on the same folds, so
   the corrected 0.5823 is still below the uncorrected uncalibrated model. Part
   of the +1.9 may be recovering what calibration cost rather than adding signal.
2. MuRIL may already represent the interaction; a transformer can and a bag of
   n-grams cannot. If it does, this adds nothing on top. That is the open
   question this module exists to answer, and it needs a 5-fold oof_probs.npy
   from the current recipe.

Costs no GPU and changes no weights: it is applied to a probability matrix.
"""
import re

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ACT = re.compile(
    r"\b(hodi|hode|hodk|odi|haak|haku|hak|kollu|kol|sayis|saayis|kole|kachis|suttu|"
    r"benki|gallige|encounter|goli|bomb|gun|war|yuddha|chappal|chapli|mettal|"
    r"kill|murder|rape|terrorist|terrist|atankav|taliban|alquida)\w*", re.I)

# Ordered: the first match wins, so a comment naming both a party and a channel is
# political. Measured precedence, not a guess -- see the table in the docstring.
CONTEXTS = [
    ("political", re.compile(r"\b(bjp|congress|congi|modi|siddu|sidda|bommai|cm|mla|mp|"
                             r"speaker|election|vote|party|govt|sarkara)\b", re.I)),
    ("religion", re.compile(r"\b(muslim\w*|hindu\w*|islam\w*|hijab|halal|allah|dharma|"
                            r"jati)\b", re.I)),
    ("geo", re.compile(r"\b(pakistan\w*|kashmir|tamil\w*|kaveri|india|karnataka|desha|"
                       r"naadu)\b", re.I)),
    ("media", re.compile(r"\b(btv|tv9|tv|channel|chanal|chennal|news|media|interview|"
                         r"anchor|reporter|trp|bucket|bigg|boss|film|movie|cinema|"
                         r"serial|show|youtube)\w*", re.I)),
]
N_CELLS = 2 * (len(CONTEXTS) + 1)
WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]


def cells(texts):
    """Cell id per comment: (has violent action word) x (which context, or none)."""
    out = []
    for t in texts:
        a = 1 if ACT.search(t) else 0
        i = next((j for j, (_, rx) in enumerate(CONTEXTS) if rx.search(t)), len(CONTEXTS))
        out.append(a * (len(CONTEXTS) + 1) + i)
    return np.array(out)


def fit_lift(cell_ids, y, n_classes=6):
    """log P(class | cell) - log P(class), add-one smoothed. Training rows only."""
    lift = np.zeros((N_CELLS, n_classes))
    g = np.bincount(y, minlength=n_classes) + 1.0
    g /= g.sum()
    for c in range(N_CELLS):
        m = cell_ids == c
        p = (np.bincount(y[m], minlength=n_classes) + 1.0) if m.any() else np.ones(n_classes)
        lift[c] = np.log(p / p.sum()) - np.log(g)
    return lift


def apply_lift(probs, cell_ids, lift, weight):
    return np.log(probs + 1e-9) + weight * lift[cell_ids]


def select_weight(texts, y, probs, seed=7, n_inner=4, weights=WEIGHTS):
    """Pick the weight on an inner split so the scored rows never choose it."""
    c = cells(texts)
    itr, iva = next(StratifiedKFold(n_inner, shuffle=True, random_state=seed).split(probs, y))
    lift = fit_lift(c[itr], y[itr], probs.shape[1])
    scores = {w: f1_score(y[iva], apply_lift(probs[iva], c[iva], lift, w).argmax(1),
                          average="macro") for w in weights}
    return max(scores, key=scores.get), scores


def nested_score(texts, y, probs, n_folds=5, seed=42, weights=WEIGHTS):
    """Honest estimate: the weight is chosen inside each fold, never on its own rows.

    Returns (macro_f1_corrected, macro_f1_plain, chosen weight per fold).
    """
    c = cells(texts)
    pred = np.zeros(len(y), dtype=int)
    chosen = []
    for tr, va in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(probs, y):
        w, _ = select_weight([texts[i] for i in tr], y[tr], probs[tr], weights=weights)
        lift = fit_lift(c[tr], y[tr], probs.shape[1])
        pred[va] = apply_lift(probs[va], c[va], lift, w).argmax(1)
        chosen.append(w)
    return (f1_score(y, pred, average="macro"),
            f1_score(y, probs.argmax(1), average="macro"), chosen)
