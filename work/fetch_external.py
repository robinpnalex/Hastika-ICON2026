"""Build an auxiliary Kannada-English hate corpus from OffensEval-Dravidian.

Source: community-datasets/offenseval_dravidian, `kannada` config (CC BY 4.0),
the DravidianCodeMix / KanCMD YouTube crawl. Same language pair, same platform,
same romanised Latin script as HASTIKA -- verified, not assumed.

Label mapping. The source has six classes; only three are used:
    Offensive_Targeted_Insult_{Individual,Group,Other}  -> Hate      (definite)
    Not_offensive                                       -> Non-Hate  (complement)
    Offensive_Untargetede                               -> DROPPED
    not-Kannada                                         -> DROPPED
Untargeted profanity is dropped because it is precisely the band where HASTIKA's
annotators may have said Non-Hate; including it imports label noise, not signal.

Ids are prefixed `ext_` so they can never collide or be joined with a HASTIKA id.
Rows whose cleaned text matches any HASTIKA comment are dropped: both corpora are
Kannada YouTube comments, so overlap is plausible, and an overlapping row that
also sits in the unreleased test split would be contamination.
"""
import argparse
import csv
import io
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep import clean  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASE = ("https://huggingface.co/datasets/community-datasets/offenseval_dravidian"
        "/resolve/refs%2Fconvert%2Fparquet/kannada")
NAMES = ["Not_offensive", "Offensive_Untargetede", "Offensive_Targeted_Insult_Individual",
         "Offensive_Targeted_Insult_Group", "Offensive_Targeted_Insult_Other", "not-Kannada"]
HATE = {"Offensive_Targeted_Insult_Individual", "Offensive_Targeted_Insult_Group",
        "Offensive_Targeted_Insult_Other"}


def fetch(split):
    import pyarrow.parquet as pq
    url = f"{BASE}/{split}/0000.parquet"
    with urllib.request.urlopen(url, timeout=180) as r:
        blob = r.read()
    t = pq.read_table(io.BytesIO(blob)).to_pydict()
    return list(zip(t["text"], t["label"]))


def hastika_texts():
    """Every HASTIKA comment we hold, for dedup. Text only -- no labels, no ids."""
    seen = set()
    for name in ["binary_train.csv", "binary_validation_inputs.csv",
                 "multiclass_train.csv", "multiclass_validation_inputs.csv"]:
        p = ROOT / "data" / name
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                seen.add(clean(row["Comment"]).casefold())
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/external/offenseval_kn.csv")
    ap.add_argument("--min-words", type=int, default=2,
                    help="drop stubs; a 1-word comment carries no usable context")
    ap.add_argument("--keep-native-script", action="store_true",
                    help="keep rows written in Kannada script. Off by default: HASTIKA is "
                         "~100%% romanised (366 Kannada chars in 396k), this source is ~35%% "
                         "native script, and that is a distribution HASTIKA never shows.")
    args = ap.parse_args()

    blocked = hastika_texts()
    rows, stats = [], {"dropped_class": 0, "dup_hastika": 0, "dup_self": 0,
                       "too_short": 0, "native_script": 0}
    emitted = set()

    for split in ["train", "validation"]:
        for text, idx in fetch(split):
            name = NAMES[idx]
            if name in HATE:
                label = "Hate"
            elif name == "Not_offensive":
                label = "Non-Hate"
            else:
                stats["dropped_class"] += 1
                continue
            c = clean(text)
            key = c.casefold()
            native = sum(1 for ch in c if "\u0c80" <= ch <= "\u0cff")
            latin = sum(1 for ch in c if ch.isascii() and ch.isalpha())
            if not args.keep_native_script and native > latin:
                stats["native_script"] += 1
            elif len(c.split()) < args.min_words:
                stats["too_short"] += 1
            elif key in blocked:
                stats["dup_hastika"] += 1
            elif key in emitted:
                stats["dup_self"] += 1
            else:
                emitted.add(key)
                rows.append((f"ext_{len(rows):05d}", text, label, name))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "Comment", "Label", "source_label"])
        w.writerows(rows)

    n_hate = sum(1 for r in rows if r[2] == "Hate")
    print(f"wrote {out}  {len(rows)} rows  Hate {n_hate}  Non-Hate {len(rows) - n_hate}")
    print("  filtered:", stats)


if __name__ == "__main__":
    main()
