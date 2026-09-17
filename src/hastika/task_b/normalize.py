"""Input-text transforms, stacked on top of the Run 9 recipe without changing it.

Each transform takes the already-cleaned comment text and returns new text. The
identity transform `raw` is what every submitted model so far has used, so an arm
that passes `--text-transform raw` is byte-for-byte the current recipe.

A WARNING THAT APPLIES TO stopwords AND stem
--------------------------------------------
The encoder these run on was adapted by masked-LM over the *raw* cleaned text
(`tapt.py`, worth +2.9 points, the largest measured gain in the project). Both of
these transforms change the classifier's input so it no longer matches what TAPT
saw. Measuring them is fair, but a loss here is ambiguous: it may be the idea, or
it may be the mismatch. To separate the two you would have to re-run TAPT on the
transformed text as well, at +45 min per arm.

Neither list below exists for romanized Kannada anywhere public, so both are
derived from this corpus rather than taken from a reference. A negative result is
therefore partly a result about these implementations.
"""
import collections
import re

WORD = re.compile(r"[A-Za-z]+")

# Kannada is agglutinative: case, number and verb inflection are suffixes stacked
# on a stem. Ordered longest-first so `-annu` strips before `-nu`. MIN_STEM stops
# the stripper eating short words whole (`nanu` -> `nan`, not `n`).
SUFFIXES = [
    "gaLannu", "gaLalli", "gaLige", "gaLinda", "gaLa", "galannu", "galalli",
    "galige", "galinda", "gala", "annu", "alli", "inda", "ige", "ali", "ade",
    "ante", "aagi", "agi", "ondu", "ide", "ittu", "utte", "uttu", "tare",
    "thare", "taare", "dare", "dhare", "kke", "ge", "na", "ru", "lu", "vu",
    "du", "da", "ni", "nu", "ke",
]
MIN_STEM = 4


def stem_word(w):
    low = w.lower()
    for s in SUFFIXES:
        if low.endswith(s) and len(low) - len(s) >= MIN_STEM:
            return low[: -len(s)]
    return low


def stem(text):
    return WORD.sub(lambda m: stem_word(m.group()), text)


def build_stoplist(texts, top_n=60):
    """The most frequent word types in this corpus.

    There is no published stoplist for romanized Kannada, and frequency is the
    only label-free definition available. Built from the training fold's text
    only, so it carries no label information.
    """
    c = collections.Counter(w.lower() for t in texts for w in WORD.findall(t))
    return {w for w, _ in c.most_common(top_n)}


def drop_stopwords(text, stoplist):
    kept = [w for w in text.split() if w.lower().strip(".,!?") not in stoplist]
    # never return an empty string: an empty input is a different experiment
    return " ".join(kept) if kept else text


def apply_transform(name, X, X_test, polarity=None):
    """Returns (X, X_test) transformed. `raw` is the identity, bit-for-bit."""
    if name == "raw":
        return X, X_test
    if name == "stem":
        return [stem(t) for t in X], [stem(t) for t in X_test]
    if name == "stopwords":
        stop = build_stoplist(list(X))
        print(f"  stoplist ({len(stop)}): {' '.join(sorted(stop)[:20])} ...", flush=True)
        return ([drop_stopwords(t, stop) for t in X],
                [drop_stopwords(t, stop) for t in X_test])
    if name == "polarity":
        if polarity is None:
            raise SystemExit("--text-transform polarity needs --polarity-map")
        tag = lambda t: f"{t} | {polarity.get(t.strip(), 'unknown polarity')}"
        return [tag(t) for t in X], [tag(t) for t in X_test]
    raise SystemExit(f"unknown transform {name}")
