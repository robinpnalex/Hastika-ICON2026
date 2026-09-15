"""One place that turns a list of data sources into a list of comments.

Every script that consumes unlabelled text -- tapt.py, extend_vocab.py -- goes
through here, so adding a new corpus to a run is a path on the command line and
nothing else. Accepts:

    data/raw/multiclass_train.csv          a CSV; the text column is auto-detected
    data/external/*.csv                a glob
    scraped/comments.txt               one comment per line
    scraped/dump.jsonl                 one JSON object per line, text auto-detected
    @experiments/task_b/corpora.txt    a manifest: one source per line, # comments

POLICY
------
Three files contain Task B test comments and are gated behind an explicit flag,
because whether they may be used is a shared-task rules question and not a
judgement call this code should make silently:

    binary_train.csv                319 of the 395 Task B test ids appear in it
    binary_validation_inputs.csv
    multiclass_validation_inputs.csv

No labels are ever read here -- masked-LM and vocabulary counting need text
alone -- so the question is purely whether the organizers permit transductive
use of released inputs. Note the repository is currently inconsistent about
this: tapt.py refused all three, while muril_b.py's build_trimmed_vocab reads
multiclass_validation_inputs.csv on the grounds that an unlabelled token
inventory leaks nothing. --allow-transductive resolves it one way, explicitly,
and prints what it admitted so the decision ends up in the run log.
"""
import glob
import pathlib
import sys

import pandas as pd

from hastika.common.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
GATED = {"binary_train.csv", "binary_validation_inputs.csv",
         "multiclass_validation_inputs.csv"}
TEXT_COLS = ["Comment", "comment", "text", "Text", "tweet", "content", "sentence"]


def _text_column(df, path):
    for c in TEXT_COLS:
        if c in df.columns:
            return c
    obj = [c for c in df.columns if df[c].dtype == object]
    if not obj:
        sys.exit(f"{path}: no text-like column in {list(df.columns)}")
    return max(obj, key=lambda c: df[c].astype(str).str.len().mean())


def _expand(spec):
    """Manifest, glob, or plain path -> concrete paths."""
    if spec.startswith("@"):
        man = pathlib.Path(spec[1:])
        if not man.is_absolute():
            man = ROOT / man
        out = []
        for line in man.read_text().splitlines():
            line = line.split("#")[0].strip()
            if line:
                out += _expand(line)
        return out
    p = spec if pathlib.Path(spec).is_absolute() else str(ROOT / spec)
    hits = sorted(glob.glob(p))
    if not hits:
        sys.exit(f"no such source: {spec}")
    return [pathlib.Path(h) for h in hits]


def _read(path):
    suf = path.suffix.lower()
    if suf == ".csv":
        df = pd.read_csv(path)
        return df[_text_column(df, path)].astype(str).tolist()
    if suf in (".jsonl", ".ndjson"):
        df = pd.read_json(path, lines=True)
        return df[_text_column(df, path)].astype(str).tolist()
    return [l for l in path.read_text(encoding="utf-8", errors="replace").splitlines()]


def load(specs, clean_fn=None, allow_transductive=False, min_words=2,
         dedupe=True, verbose=True):
    """Resolve sources to a deduplicated list of comments. Reads no labels."""
    texts, admitted = [], []
    for spec in specs:
        for path in _expand(spec):
            if path.name in GATED:
                if not allow_transductive:
                    sys.exit(f"refusing {path.name}: it carries Task B test comments. "
                             f"Pass --allow-transductive to use it deliberately; see "
                             f"the POLICY note in hastika.task_b.corpora.")
                admitted.append(path.name)
            rows = _read(path)
            if clean_fn:
                rows = [clean_fn(t) for t in rows]
            rows = [t for t in rows if len(t.split()) >= min_words]
            if verbose:
                print(f"  {path.relative_to(ROOT) if ROOT in path.parents else path}"
                      f": {len(rows)} rows", flush=True)
            texts += rows
    if admitted:
        print(f"  TRANSDUCTIVE: admitted {sorted(set(admitted))} by explicit flag. "
              f"No labels were read. Confirm the shared task permits this.", flush=True)
    n_raw = len(texts)
    if dedupe:
        seen, out = set(), []
        for t in texts:                    # a repeated comment is not extra evidence
            k = t.casefold()
            if k not in seen:
                seen.add(k)
                out.append(t)
        texts = out
    if verbose:
        print(f"corpus: {len(texts)} unique comments (from {n_raw})", flush=True)
    return texts
