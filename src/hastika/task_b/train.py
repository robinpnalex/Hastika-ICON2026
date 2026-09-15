"""MuRIL fine-tune for HASTIKA Task B (6-way hate category).

Reuses every piece of the tuned Task A recipe in muril.py -- meanmax pooling,
LLRD, top-layer re-init, FGM, EMA, multi-seed, twice-per-epoch checkpoint
selection on macro-F1 -- and changes only what Task B actually needs:

  * 6 output classes instead of 2.
  * Class-weighted cross-entropy. Task A was near-balanced so muril.py weights
    nothing; Task B is 1362 Gender vs 186 Geo-political, and macro-F1 counts
    those equally, so the tail has to be weighted up.
  * argmax decode. muril.py tunes a p(Hate) threshold; the 6-class analogue
    (per-class score offsets) was measured on the TF-IDF OOF and moved macro-F1
    by +0.002, inside noise, because class weighting already does that job.

Split seed and fold count match baseline_svm.py so the OOF rows line up and
ensemble.py can blend the two.

--trim-vocab is optional and off by default: it drops the embedding rows for the
95% of MuRIL's 197k wordpieces this corpus never emits (293M params -> 93M). It
exists because MuRIL OOM-killed a 6GB CPU box at 10.5GB peak RSS; on a GPU there
is no reason to modify the model. Coverage was measured before relying on it --
holding out an entire released CSV, 0.178% of its tokens fall outside a vocab
built from the other three, because wordpiece backs off to seen subwords.
"""
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import AutoTokenizer

from hastika.common.paths import RAW_DATA_DIR
from hastika.common.preprocessing import clean, dedupe_index
from hastika.models.muril import N_SPLITS, ROOT, RUNS, SPLIT_SEED, train_fold
from .features import TagLexicon, describe

LABELS = ["Gender", "Geo-political", "Others", "Political", "Religion", "Violence"]


def build_trimmed_vocab(tok, max_len, demoji):
    """Restrict MuRIL's 197k-row embedding table to the tokens this corpus uses.

    Built from all four released CSVs, not just the training split -- the test
    inputs are unlabelled, so reading their token inventory leaks nothing. Peak
    RSS on this 6GB box drops from an OOM kill to a run that fits; see
    _trim_vocab in muril.py for the coverage measurement.
    """
    keep = set(tok.all_special_ids)
    for name in ["binary_train.csv", "binary_validation_inputs.csv",
                 "multiclass_train.csv", "multiclass_validation_inputs.csv"]:
        for t in pd.read_csv(RAW_DATA_DIR / name)["Comment"]:
            keep.update(tok(clean(t, demojize=demoji), truncation=True,
                            max_length=max_len)["input_ids"])
    keep_ids = torch.tensor(sorted(keep), dtype=torch.long)
    remap = torch.full((tok.vocab_size,), -1, dtype=torch.long)
    remap[keep_ids] = torch.arange(len(keep_ids))
    remap[remap == -1] = int((keep_ids == tok.unk_token_id).nonzero())
    print(f"vocab trimmed to {len(keep_ids)} of {tok.vocab_size} tokens "
          f"({len(keep_ids)/tok.vocab_size*100:.1f}%)", flush=True)
    return keep_ids, remap


def tag_split(args, X, y, tr_i, va_i, X_test, verbose=False):
    """Fit the tag lexicon on this fold's TRAINING rows, then tag all three sets.

    Fitting before the split would put the validation rows' own labels into their
    input text and inflate OOF. The test inputs are tagged with the same fold-local
    lexicon, which is why the tagged test set is rebuilt per fold rather than once.
    """
    if not args.tags:
        return X[tr_i], X[va_i], X_test
    lex = TagLexicon(LABELS, top_k=args.tag_top_k,
                     min_z=args.tag_min_z).fit(X[tr_i], y[tr_i])
    if verbose:
        print("  fold gazetteer (first 8 terms per class):", flush=True)
        print(describe(lex), flush=True)
    return lex.transform(X[tr_i]), lex.transform(X[va_i]), lex.transform(X_test)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/muril-base-cased")
    ap.add_argument("--tag", default="muril_b")
    ap.add_argument("--no-dedupe", action="store_true",
                    help="keep repeated comments and label-conflict groups (prep.dedupe_index)")
    ap.add_argument("--max-len", type=int, default=192,
                    help="192, not muril.py's 128: Task B's comments are longer than Task "
                         "A's. Measured on MuRIL wordpieces, p99 is 145 tokens in "
                         "multiclass_train and 163 in multiclass_validation_inputs, so 128 "
                         "truncates 1.4%% of training and 2.0%% of test rows against 0.5%% "
                         "at 192. Padding is per batch, so the longer cap is only paid by "
                         "the rare long batch")
    ap.add_argument("--no-demojize", action="store_true")
    ap.add_argument("--folds", type=int, default=N_SPLITS,
                    help="0 = 15%% holdout, 1 = fit every row with no "
                         "validation and no score, >1 = k-fold OOF")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--eval-bs", type=int, default=64)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--evals-per-epoch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--llrd", type=float, default=0.9)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--pooling", choices=["meanmax", "mean", "cls", "last4"], default="meanmax")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--reinit-layers", type=int, default=2)
    ap.add_argument("--fgm", action="store_true", default=True)
    ap.add_argument("--no-fgm", dest="fgm", action="store_false")
    ap.add_argument("--fgm-eps", type=float, default=1.0)
    ap.add_argument("--ema", action="store_true", default=True)
    ap.add_argument("--no-ema", dest="ema", action="store_false")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--no-ema-bias-correct", dest="ema_bias_correct", action="store_false",
                    default=True, help="restore the pre-fix EMA; for reproducing old runs only")
    ap.add_argument("--select", choices=["best", "last"], default="best",
                    help="checkpoint that supplies each fold's predictions; both scores are "
                         "reported either way, see the OOF lines at the end")
    ap.add_argument("--rdrop", type=float, default=0.0)
    ap.add_argument("--aux-weight", type=float, default=0.0,
                    help="weight on an auxiliary head predicting whether the "
                         "comment contains violent language (hastika.task_b.axes). 0 "
                         "disables it. The act axis has 443 positives against "
                         "the Violence class's 221, so this gives the encoder "
                         "twice the signal for the concept Violence depends on")
    ap.add_argument("--amp", choices=["auto", "off", "fp16", "bf16"], default="auto")
    ap.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    ap.add_argument("--loss", choices=["ce", "focal"], default="ce",
                    help="focal down-weights already-correct rows, aimed at the 17%% with "
                         "competing topic cues; composes with --class-weight balanced. "
                         "UNMEASURED on MuRIL -- A/B it with experiments/task_b/sweep.sh")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--tags", action="store_true", default=False,
                    help="append topic/mood/address tags to each comment (features.py). "
                         "OFF by default: on the TF-IDF SVM the tags moved 5-fold macro-F1 "
                         "by -0.007, inside noise, because that model already has the "
                         "gazetteer words as features. The case for MuRIL is different -- "
                         "it sees `bommai` 19 times as wordpieces and cannot learn it names "
                         "a politician -- but that case is untested. A/B it, do not assume")
    ap.add_argument("--no-tags", dest="tags", action="store_false")
    ap.add_argument("--tag-top-k", type=int, default=40,
                    help="gazetteer size per class, fitted per fold on training rows only")
    ap.add_argument("--tag-min-z", type=float, default=3.0,
                    help="min log-odds z for a gazetteer term. Swept 5-fold: 0/2/3/4 give "
                         "mean held-out tag lift 2.8/4.7/5.5/5.5 at 80/68/55/49%% coverage. "
                         "3.0 is the knee; 4.0 loses Violence entirely")
    ap.add_argument("--trim-vocab", action="store_true",
                    help="drop the 95%% of MuRIL embedding rows this corpus never emits; "
                         "saves ~2.3GB and was needed to fit a 6GB CPU box, but the bare "
                         "run keeps the full table so the model is unmodified")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="train on only the first N rows -- smoke-testing a new machine")
    args = ap.parse_args()
    args.n_classes = len(LABELS)

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "-"
    print(f"device={device} ({gpu}) model={args.model} pooling={args.pooling} "
          f"fgm={args.fgm} ema={args.ema} reinit={args.reinit_layers} "
          f"class_weight={args.class_weight} loss={args.loss} tags={args.tags} "
          f"seeds={args.seeds}", flush=True)

    train = pd.read_csv(RAW_DATA_DIR / "multiclass_train.csv")
    test = pd.read_csv(RAW_DATA_DIR / "multiclass_validation_inputs.csv")
    if not args.no_dedupe:
        train = train.iloc[dedupe_index(train["Comment"].tolist(),
                                        train["Hate Category"].tolist(), "task B")].reset_index(drop=True)
    demoji = not args.no_demojize
    X = train["Comment"].map(lambda t: clean(t, demojize=demoji)).values
    y = train["Hate Category"].map(LABELS.index).values
    X_test = test["Comment"].map(lambda t: clean(t, demojize=demoji)).values
    assert not pd.isna(y).any(), "unmapped category label"

    if args.limit:
        X, y = X[:args.limit], y[:args.limit]
        print(f"LIMIT: using {len(y)} training rows (smoke test)", flush=True)

    if args.class_weight == "balanced":
        counts = np.bincount(y, minlength=len(LABELS))
        w = len(y) / (len(LABELS) * np.maximum(counts, 1))
        args.class_weight_t = torch.tensor(w, dtype=torch.float, device=device)
        print("class weights:", dict(zip(LABELS, w.round(3))), flush=True)
    else:
        args.class_weight_t = None

    tok = AutoTokenizer.from_pretrained(args.model)
    if args.trim_vocab:
        args.keep_ids, args.remap = build_trimmed_vocab(tok, args.max_len, demoji)

    run = RUNS / args.tag
    run.mkdir(parents=True, exist_ok=True)

    oof = np.zeros((len(y), len(LABELS)))
    test_probs = np.zeros((len(X_test), len(LABELS)))
    # Same two matrices for the checkpoint --select did not take, so one run
    # reports both the selected and the unselected OOF.
    oof_alt = np.zeros_like(oof)
    test_alt = np.zeros_like(test_probs)
    n_seeds = len(args.seeds)

    for seed in args.seeds:
        args.seed = seed
        if args.folds and args.folds > 1:
            skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SPLIT_SEED)
            for k, (tr_i, va_i) in enumerate(skf.split(X, y), 1):
                print(f"===== seed {seed} fold {k}/{args.folds} =====", flush=True)
                Xtr, Xva, Xte = tag_split(args, X, y, tr_i, va_i, X_test, verbose=(k == 1))
                _, p_va, p_te = train_fold(args, tok, Xtr, y[tr_i], Xva, y[va_i],
                                           Xte, device, f"s{seed}f{k}")
                oof[va_i] += p_va / n_seeds
                test_probs += p_te / (args.folds * n_seeds)
                alt = args.alt_fold
                if alt["va"] is not None:
                    oof_alt[va_i] += alt["va"] / n_seeds
                    test_alt += alt["te"] / (args.folds * n_seeds)
        elif args.folds == 1:
            # Train on every labelled row. No held-out rows means no score: the
            # run is unmeasurable by construction and prints nothing to compare.
            print(f"===== seed {seed} FULL FIT, {len(y)} rows, no validation =====",
                  flush=True)
            Xtr, _, Xte = tag_split(args, X, y, np.arange(len(y)),
                                    np.arange(0), X_test, verbose=True)
            _, _, p_te = train_fold(args, tok, Xtr, y, np.array([], dtype=Xtr.dtype),
                                    np.array([], dtype=y.dtype), Xte, device,
                                    f"s{seed}full")
            test_probs += p_te / n_seeds
        else:
            tr_i, va_i = train_test_split(np.arange(len(y)), test_size=0.15,
                                          stratify=y, random_state=SPLIT_SEED)
            Xtr, Xva, Xte = tag_split(args, X, y, tr_i, va_i, X_test, verbose=True)
            _, p_va, p_te = train_fold(args, tok, Xtr, y[tr_i], Xva, y[va_i],
                                       Xte, device, f"s{seed}holdout")
            oof[va_i] += p_va / n_seeds
            test_probs += p_te / n_seeds
            alt = args.alt_fold
            if alt["va"] is not None:
                oof_alt[va_i] += alt["va"] / n_seeds
                test_alt += alt["te"] / n_seeds

    other = "last" if args.select == "best" else "best"
    if args.folds == 1:
        print(f"\nfull fit on {len(y)} rows. There is no score for this run and there "
              f"cannot be one:\nevery labelled row was used for training. Compare it "
              f"against a 5-fold run's\nOOF only by submitting both.")
        print("\npredicted distribution on the test inputs vs the training prior:")
        got = np.bincount(test_probs.argmax(1), minlength=len(LABELS))
        prior = np.bincount(y, minlength=len(LABELS))
        print(f"  {'class':16s} {'predicted':>10s} {'training':>10s}")
        for i, l in enumerate(LABELS):
            print(f"  {l:16s} {100*got[i]/got.sum():9.1f}% {100*prior[i]/prior.sum():9.1f}%")
    elif args.folds and args.folds > 1:
        pred = oof.argmax(1)
        print(f"\nOOF macro-F1 {f1_score(y, pred, average='macro'):.4f} "
              f"acc {accuracy_score(y, pred):.4f}  (--select {args.select})")
        if oof_alt.any():
            print(f"OOF macro-F1 {f1_score(y, oof_alt.argmax(1), average='macro'):.4f} "
                  f"with the {other} checkpoint instead. Trust the 'last' number: "
                  f"'best' is chosen on these same rows.\n")
        else:
            print()
        print(classification_report(y, pred, target_names=LABELS, digits=3))
        print("confusion (rows=true, cols=pred):")
        print("%-15s" % "", " ".join("%6s" % l[:6] for l in LABELS))
        for l, row in zip(LABELS, confusion_matrix(y, pred)):
            print("%-15s" % l, " ".join("%6d" % v for v in row))
        np.save(run / "oof_probs.npy", oof)
    else:
        mask = oof.any(1)
        pred = oof[mask].argmax(1)
        print(f"\nholdout macro-F1 {f1_score(y[mask], pred, average='macro'):.4f} "
              f"acc {accuracy_score(y[mask], pred):.4f}\n")
        print(classification_report(y[mask], pred, target_names=LABELS, digits=3))
        if oof_alt.any():
            m2 = oof_alt.any(1)
            print(f"holdout macro-F1 {f1_score(y[m2], oof_alt[m2].argmax(1), average='macro'):.4f} "
                  f"with the {other} checkpoint instead")
        np.save(run / "holdout_probs.npy", oof)

    np.save(run / "test_probs.npy", test_probs)
    if test_alt.any():
        np.save(run / f"test_probs_{other}.npy", test_alt)
    pd.DataFrame({"id": test["id"],
                  "label": [LABELS[i] for i in test_probs.argmax(1)]}
                 ).to_csv(run / "predictions.csv", index=False)
    print(f"wrote {run}/predictions.csv")


if __name__ == "__main__":
    main()
