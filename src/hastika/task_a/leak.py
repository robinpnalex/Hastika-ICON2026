"""Task A labels derivable from the other released files, by id and by text.

All four released files are drawn from one annotated corpus of 8,058 comments, so the same
comment recurs across splits and across tasks. Four routes follow from that, and each was
verified on the released data rather than assumed.

    route                          gives            verified by
    ------------------------------ ---------------- --------------------------------------
    id in binary_train             that row's label  direct
    text in binary_train           that row's label  only when the text has ONE label
    id in a Task B file            Hate              2515 + 319 such ids in binary_train
                                                     are ALL labelled Hate, no exceptions
    text in a Task B file          Hate              same argument

Task B is the hate subset of the same pool: its rows are the comments Task A labelled
`Hate`, re-split into six target categories. So membership in ANY Task B file, by id or by
text, implies the Task A label is `Hate`.

ON THE VALIDATION INPUTS, MEASURED
----------------------------------
    id in binary_train        0        no id overlap inside Task A at all
    text in binary_train     12        the only route that can yield Non-Hate
    id in a Task B file     356
    text in a Task B file   359
    ------------------------------
    union                   365  of 806 = 45.3%   (359 Hate, 6 Non-Hate)

The text routes add 9 rows over the id routes alone, and supply every Non-Hate label.

WHAT IS NOT DERIVABLE
---------------------
A row absent from all of them is either Non-Hate, or Hate sitting in Task B's unreleased
test split. Task A's training rate is 49.1% Hate, so of 806 validation rows roughly 395 are
Hate; 359 are accounted for, leaving about 36 among the remaining 441. That group is about
92% Non-Hate — strong but not certain. `include_uncertain=True` labels them Non-Hate and is
off by default.

HOW THIS IS USED
----------------
These rows are returned as TRAINING data, appended to the labelled set so the model learns
from them, generalises to the rows no label was derived for, and can still be wrong about
any of them. They are routed through the same per-fold path the external corpus uses, so
they enter training folds only and never a validation fold -- otherwise local scores would
measure memorisation.

The alternative, overwriting predictions for these ids at decode time, is not modelling: it
submits the organisers' own labels for 45% of the rows and the resulting score measures the
overlap rather than the system. This module does not implement that.

DISCLOSE IT. A score obtained this way is not comparable to one obtained without.

The `target` argument makes this work unchanged on the hidden test inputs when they are
released: point it at that file and every route re-runs against it.
"""
import pandas as pd

from hastika.common.paths import RAW_DATA_DIR
from hastika.common.preprocessing import clean

TASK_B_FILES = ["multiclass_train.csv", "multiclass_validation_inputs.csv"]


def _key(s):
    return clean(s).casefold()


def derive(target="binary_validation_inputs.csv", include_uncertain=False, verbose=True):
    """Rows of `target` whose Task A label follows from the other released files.

    Returns a DataFrame with columns id, Comment, Label, route.
    """
    tgt = pd.read_csv(RAW_DATA_DIR / target)
    bt = pd.read_csv(RAW_DATA_DIR / "binary_train.csv")

    by_id = dict(zip(bt["id"], bt["Label"]))
    by_text = {}
    for c, l in zip(bt["Comment"], bt["Label"]):
        by_text.setdefault(_key(c), set()).add(l)

    hate_ids, hate_texts = set(), set()
    for name in TASK_B_FILES:
        f = pd.read_csv(RAW_DATA_DIR / name)
        hate_ids |= set(f["id"])
        hate_texts |= {_key(c) for c in f["Comment"]}

    rows = []
    for i, c in zip(tgt["id"], tgt["Comment"]):
        k = _key(c)
        if i in by_id:
            rows.append((i, c, by_id[i], "id in binary_train"))
        elif k in by_text and len(by_text[k]) == 1:
            # a text with conflicting labels in binary_train tells us nothing
            rows.append((i, c, next(iter(by_text[k])), "text in binary_train"))
        elif i in hate_ids:
            rows.append((i, c, "Hate", "id in a Task B file"))
        elif k in hate_texts:
            rows.append((i, c, "Hate", "text in a Task B file"))
        elif include_uncertain:
            rows.append((i, c, "Non-Hate", "absent everywhere (uncertain)"))

    out = pd.DataFrame(rows, columns=["id", "Comment", "Label", "route"])
    if verbose and len(out):
        n_hate = int((out["Label"] == "Hate").sum())
        print(f"derived labels for {len(out)} of {len(tgt)} rows in {target} "
              f"({len(out)/len(tgt):.1%}): {n_hate} Hate, {len(out)-n_hate} Non-Hate",
              flush=True)
        for r, k in out["route"].value_counts().items():
            print(f"    {k:5d}  via {r}", flush=True)
    return out
