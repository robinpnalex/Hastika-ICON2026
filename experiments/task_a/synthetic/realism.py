"""Inject the surface variation that LLM-generated Kanglish systematically lacks.

Measured on batch1 vs real comments (equal-size samples):

                 synthetic   real
  3+ char repeats     0.0%  22.7%
  uppercase ratio     1.8%   6.5%
  emoji rows          8.0%  13.3%
  punctuation/comment 0.86   1.99

These are real properties of the corpus, not cosmetic noise: people do elongate
("sooo"), shout, and repeat emoji. Romanisation jitter is likewise real -- the
corpus has 70.4% hapax types largely because the same word is spelled several
ways, which a generator spelling each word consistently will never reproduce.

CAVEAT: lowering the real-vs-synthetic AUC by adding surface noise is not the
same as improving downstream transfer. Verify with evaluate.py's augmentation
number, not with the AUC alone.
"""
import random
import re

# Attested alternations in the corpus: same word, several spellings.
JITTER = [
    (r"\bnaanu\b", ["nanu", "nan", "naanu"]),
    (r"\bnalli\b", ["alli", "li", "nalli"]),
    (r"\benu\b", ["en", "enu", "enh"]),
    (r"\banta\b", ["antha", "anta", "anth"]),
    (r"\billa\b", ["illa", "ila", "illva"]),
    (r"\bmadtha\b", ["madta", "madtha", "maadtha"]),
    (r"\bavaru\b", ["avru", "avaru"]),
    (r"\bivaga\b", ["ivag", "eega", "ivaga"]),
    (r"\bchennagide\b", ["chennagide", "chnagide", "channagide"]),
    (r"\byaru\b", ["yaru", "yar", "yaaru"]),
    (r"\bbeku\b", ["beku", "bek", "bekku"]),
    (r"\bgottu\b", ["gottu", "gotu", "gothu"]),
]
EMOJI = ["😂", "🤣", "🔥", "❤️", "🙏", "💪", "😡", "😭", "😅", "👏"]


def _elongate(s, rng):
    words = s.split()
    if not words:
        return s
    i = rng.randrange(len(words))
    w = words[i]
    if len(w) > 2 and w[-1].isalpha():
        words[i] = w + w[-1] * rng.randint(2, 4)
    return " ".join(words)


def _shout(s, rng):
    words = s.split()
    cand = [i for i, w in enumerate(words) if w.isalpha() and len(w) > 3]
    if not cand:
        return s
    i = rng.choice(cand)
    words[i] = words[i].upper()
    return " ".join(words)


def realise(text, rng):
    """Apply corpus-frequency-matched surface variation to one comment."""
    s = text
    for pat, opts in JITTER:
        if re.search(pat, s) and rng.random() < 0.6:
            s = re.sub(pat, rng.choice(opts), s, count=1)
    if rng.random() < 0.23:                       # match measured 22.7%
        s = _elongate(s, rng)
    if rng.random() < 0.12:
        s = _shout(s, rng)
    if rng.random() < 0.14:                       # emoji, often repeated
        s = s.rstrip() + " " + rng.choice(EMOJI) * rng.randint(1, 4)
    if rng.random() < 0.20:
        s = s.rstrip(" .") + rng.choice(["..", "...", "?", "!", " ..", ""])
    return s


def realise_batch(rows, seed=0):
    rng = random.Random(seed)
    out = []
    for r in rows:
        r = dict(r)
        r["text"] = realise(r["text"], rng)
        out.append(r)
    return out


if __name__ == "__main__":
    import json
    import pathlib
    import sys
    src = pathlib.Path(sys.argv[1])
    dst = pathlib.Path(sys.argv[2])
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    with open(dst, "w", encoding="utf-8") as f:
        for r in realise_batch(rows, seed=13):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {dst} ({len(rows)} rows)")
