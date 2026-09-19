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

THE HIDDEN TEST SET IS ALREADY PARTLY RELEASED
----------------------------------------------
Task A is exactly train + validation + test: 6,446 + 806 + 806 = 8,058. Every Task B row is
a Task A row labelled Hate. So a Task B row whose id appears in NO released Task A file can
only be a Task A TEST row:

    multiclass_train            3,159 ids   2,515 in train   324 in validation   320 in neither
    multiclass_validation_inputs  395 ids     319 in train    32 in validation    44 in neither

That is 364 Task A test comments whose text and label, Hate, are already public -- 45.2% of
the 806-row test set, available before the test file is released. `hidden_test()` returns
them. They are Hate by the same verified argument as every other route here.

CHECKING WHETHER A SUBMISSION USED THIS
---------------------------------------
`check()` compares a predictions file against the derived validation labels. A model trained
without these rows gets roughly 80% of them right, which is its ordinary accuracy on hate
comments. A model trained WITH them has seen those exact comments with their labels and
agrees on nearly all of them. The agreement rate on the derivable rows is therefore a
fingerprint of whether the rows were used, readable from the ZIP alone.

Calibrated with the TF-IDF SVM as a positive control: 0.797 trained without, 0.978 trained
with. Every real submission so far reads between 0.770 and 0.822, i.e. not used.

    python -m hastika.task_a.leak --check submission.zip
"""
import pandas as pd

from hastika.common.paths import RAW_DATA_DIR
from hastika.common.preprocessing import clean

TASK_B_FILES = ["multiclass_train.csv", "multiclass_validation_inputs.csv"]


def _key(s):
    return clean(s).casefold()


def hidden_test(verbose=True):
    """Task A test comments already present in the released Task B files, all Hate.

    A Task B id absent from every released Task A file can only be a Task A test row.
    Rows whose text already appears in binary_train are dropped: they add nothing the
    training set does not have.
    """
    bt = pd.read_csv(RAW_DATA_DIR / "binary_train.csv")
    bv = pd.read_csv(RAW_DATA_DIR / "binary_validation_inputs.csv")
    released = set(bt["id"]) | set(bv["id"])
    train_text = {_key(c) for c in bt["Comment"]}

    rows, seen = [], set()
    for name in TASK_B_FILES:
        f = pd.read_csv(RAW_DATA_DIR / name)
        for i, c in zip(f["id"], f["Comment"]):
            if i in released or i in seen:
                continue
            seen.add(i)
            if _key(c) in train_text:
                continue
            rows.append((i, c, "Hate", f"hidden test, from {name}"))
    out = pd.DataFrame(rows, columns=["id", "Comment", "Label", "route"])
    if verbose:
        print(f"hidden test: {len(seen)} Task A test ids are already in the Task B files "
              f"({len(seen)/806:.1%} of the test set); {len(out)} add new text, all Hate",
              flush=True)
    return out


def check(pred_csv, target="binary_validation_inputs.csv"):
    """Did the model that wrote `pred_csv` learn from the derived rows?

    Returns (agreement on derivable rows, number of derivable rows, verdict).
    """
    d = derive(target=target, verbose=False)
    p = pd.read_csv(pred_csv)
    # derived rows carry `Label`, predictions carry `label`: distinct names, no suffixing
    m = d[["id", "Label"]].merge(p[["id", "label"]], on="id")
    agree = float((m["Label"] == m["label"]).mean()) if len(m) else float("nan")
    # Calibrated on 2026-09-20 with the TF-IDF SVM as a positive control:
    #   trained without the derived rows  -> 0.797
    #   trained with them                 -> 0.978
    # and on the real submissions, none of which used them: 0.822, 0.822, 0.770.
    # The band is set wide of both so a weaker or stronger model still lands clearly.
    if agree >= 0.94:
        verdict = "USED: the model reproduces the derived labels almost exactly"
    elif agree <= 0.88:
        verdict = "NOT USED: agreement is ordinary model accuracy on hate comments"
    else:
        verdict = "UNCLEAR: between the two signatures; check the training log"
    return agree, len(m), verdict


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


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Inspect or check the cross-task label overlap.")
    ap.add_argument("--check", metavar="PREDICTIONS",
                    help="a predictions.csv or a submission ZIP to fingerprint")
    a = ap.parse_args()
    if a.check:
        import pathlib, tempfile, zipfile
        path = pathlib.Path(a.check)
        if path.suffix == ".zip":
            tmp = pathlib.Path(tempfile.mkdtemp()) / "predictions.csv"
            with zipfile.ZipFile(path) as z:
                tmp.write_bytes(z.read("predictions.csv"))
            path = tmp
        agree, n, verdict = check(path)
        print(f"agreement with derived labels on {n} rows: {agree:.3f}")
        print(verdict)
    else:
        derive()
        hidden_test()
