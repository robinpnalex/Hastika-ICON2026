"""Add this corpus's frequent word-forms to MuRIL's tokenizer, then re-embed.

THE MEASUREMENT THIS EXISTS FOR
-------------------------------
MuRIL ships 197,258 subwords for 17 languages in native scripts. This corpus,
train and test together, ever emits 7,078 of them -- 3.6%. What it gets in
exchange is 2.17 wordpieces per whitespace word, and 9.9% of words shattered
into four or more pieces. The vocabulary is the wrong shape for romanized
Kannada, and a word split five ways cannot accumulate an embedding.

Adding a word as a token and initializing its embedding as the mean of the
pieces it currently decomposes into is the standard fix (it starts the new token
where the model already is, so nothing is lost) and the adaptation pass then
moves it somewhere better. Run tapt.py on the output of this script.

THE HONEST LIMIT
----------------
72% of this corpus's word types appear exactly once. A token seen once cannot
learn an embedding, so those are not candidates and the fragmentation on rare
words is untouched. This helps the frequent tail only, which is why --min-freq
defaults to 5 rather than 2.
"""
import argparse
import re
import sys
from collections import Counter

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from hastika.common.paths import PROJECT_ROOT
from hastika.common.preprocessing import clean
from . import corpora

ROOT = PROJECT_ROOT
DEFAULT_CORPUS = ["data/raw/multiclass_train.csv", "data/external/offenseval_kn.csv"]
WORD = re.compile(r"^[A-Za-z][A-Za-z']*$")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/muril-base-cased")
    ap.add_argument("--out", default="artifacts/runs/muril-extended")
    ap.add_argument("--corpus", nargs="*", default=None)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--allow-transductive", action="store_true")
    ap.add_argument("--min-freq", type=int, default=5,
                    help="a token seen fewer times than this cannot learn an embedding")
    ap.add_argument("--min-pieces", type=int, default=2,
                    help="only add words the tokenizer currently splits at least this far")
    ap.add_argument("--max-new", type=int, default=4000)
    ap.add_argument("--no-demojize", action="store_true")
    ap.add_argument("--lowercase-too", action="store_true",
                    help="also add the lowercased form when it differs")
    a = ap.parse_args()

    demoji = not a.no_demojize
    texts = corpora.load((a.corpus or DEFAULT_CORPUS) + a.extra,
                         clean_fn=lambda t: clean(t, demojize=demoji),
                         allow_transductive=a.allow_transductive)

    tok = AutoTokenizer.from_pretrained(a.model)
    freq = Counter(w for t in texts for w in t.split() if WORD.match(w))
    print(f"\n{len(freq)} word types, "
          f"{100*sum(1 for w, n in freq.items() if n == 1)/max(1,len(freq)):.0f}% appear once")

    cand = []
    for w, n in freq.most_common():
        if n < a.min_freq:
            break
        if w in tok.get_vocab():
            continue
        pieces = tok.tokenize(w)
        if len(pieces) >= a.min_pieces:
            cand.append((w, n, pieces))
        if a.lowercase_too and w.lower() != w and w.lower() not in tok.get_vocab():
            lp = tok.tokenize(w.lower())
            if len(lp) >= a.min_pieces:
                cand.append((w.lower(), n, lp))
    cand = cand[:a.max_new]
    if not cand:
        sys.exit("nothing to add: try a lower --min-freq")

    before = sum(len(tok.tokenize(w)) * n for w, n in freq.items())
    total_words = sum(freq.values())
    print(f"adding {len(cand)} tokens (freq {cand[-1][1]}..{cand[0][1]})")
    print("examples:")
    for w, n, p in cand[:8]:
        print(f"  {w:22s} x{n:<5d} {p}")

    model = AutoModelForMaskedLM.from_pretrained(a.model)
    n_added = tok.add_tokens([w for w, _, _ in cand])
    model.resize_token_embeddings(len(tok))
    emb = model.get_input_embeddings().weight.data
    with torch.no_grad():
        for w, _, pieces in cand:
            ids = tok.convert_tokens_to_ids(pieces)
            ids = [i for i in ids if i is not None and i < emb.shape[0]]
            if ids:
                # start the new token exactly where the model already represents it
                emb[tok.convert_tokens_to_ids(w)] = emb[ids].mean(0)

    after = sum(len(tok.tokenize(w)) * n for w, n in freq.items())
    print(f"\nwordpieces per word  {before/total_words:.2f} -> {after/total_words:.2f}"
          f"   ({100*(before-after)/before:.1f}% fewer pieces)")
    print(f"vocabulary {len(tok) - n_added:,} -> {len(tok):,}")

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"\nwrote {out}")
    print(f"next:  python -m hastika.task_b.tapt --model {a.out} "
          "--out artifacts/runs/tapt-extended")
    print("then:  python -m hastika.task_b.train "
          "--model artifacts/runs/tapt-extended --tag b_ext")


if __name__ == "__main__":
    main()
