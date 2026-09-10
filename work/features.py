"""Feature tags appended to the comment text before tokenization.

Measured on multiclass_train, three properties of this corpus are not things a
3,143-row fine-tune can learn on its own, but are trivial to hand the model as
extra words:

  * Task B is largely TOPIC classification. Political, Religion and Geo-political
    are decided by closed sets of proper nouns -- bjp/congress/bommai, muslim/
    hindu/hijab, pakistan/kaveri/kashmir. A gazetteer built from the training
    fold gives 4-9x lift over the class base rate; MuRIL sees `bommai` 19 times
    and cannot learn it names a politician.
  * Violence is not a topic, it is a SPEECH ACT. Its marker is the Kannada
    necessitative suffix -beku ("must/should") on a violent stem: sayisbeku,
    hodibeku, odibeku, hakbeku. The suffix alone lifts P(Violence) from 0.070 to
    0.184. Neither stem nor mood suffices; the model needs the conjunction, which
    is what a 221-row class cannot teach.
  * Hate here is grammatically SECOND PERSON. nin/ninna/ninge/magane run 208 to
    51 against the Non-Hate half, while non-hate uses sir/jai/bro/anna/guru. Both
    registers are vocative-heavy and the vocative's register is the signal.

Tags are emitted as ordinary English words, not symbols: MuRIL has real
embeddings for "political topic" and none for "<TOPIC_POL>".

IMPORTANT. The topic gazetteer is fitted from labels, so it MUST be fitted on the
training rows of a fold and only then applied to that fold's validation rows.
Fitting on all of X before the split leaks the validation labels into the input
and inflates OOF. TagLexicon.fit is therefore called inside the fold loop.
The mood and address lexicons are linguistic, use no labels, and are constant.
"""
import collections
import math
import re

WORD = re.compile(r"[a-z]+")

# --- label-free lexicons -------------------------------------------------

# Kannada 2nd-person pronoun/possessive forms and familiar vocatives. Being
# addressed is what distinguishes abuse from commentary in this corpus.
SECOND_PERSON = {
    "nin", "ninn", "ninna", "ninage", "ninge", "ninu", "neenu", "nee", "nim",
    "nimma", "nimge", "nivu", "neevu", "ninnna", "ninnu", "nindu", "ninmakke",
}
RESPECTFUL = {
    "sir", "madam", "mam", "anna", "akka", "bro", "guru", "sara", "swamy",
    "sr", "ji", "avare", "avaru", "thamma", "jai",
}
# necessitative -beku / -bek: "must", "should". Marks a demand or an incitement.
NECESSITATIVE = re.compile(r"\b\w{3,}be?ku\b")

TAG_TOPIC = {
    "Gender": "gendered abuse topic",
    "Political": "political topic",
    "Religion": "religious topic",
    "Geo-political": "regional territory topic",
    "Violence": "violent act topic",
    "Others": "media channel topic",
}


def _types(text):
    return set(WORD.findall(text.lower()))


class TagLexicon:
    """Per-class gazetteers fitted by log-odds on one fold's training rows.

    top_k terms per class, each needing min_count occurrences overall, scored by
    the log-odds ratio with an informative Dirichlet prior (Monroe et al. 2008).
    That prior is what stops rare-but-lopsided terms from dominating, which
    matters at 184 rows for Geo-political.
    """

    def __init__(self, label_names, top_k=40, min_count=5, min_z=3.0):
        self.label_names = list(label_names)
        self.top_k, self.min_count, self.min_z = top_k, min_count, min_z
        self.gaz = {}

    def fit(self, texts, labels):
        docs = [_types(t) for t in texts]
        for ci, cname in enumerate(self.label_names):
            inside, outside = collections.Counter(), collections.Counter()
            for d, l in zip(docs, labels):
                (inside if l == ci else outside).update(d)
            total = inside + outside
            n_in, n_out = sum(inside.values()), sum(outside.values())
            N = n_in + n_out
            scored = []
            for w, c in total.items():
                if c < self.min_count or len(w) < 3:
                    continue
                a_in, a_out = inside[w] + c, outside[w] + c
                delta = (math.log(a_in / (n_in + N - a_in))
                         - math.log(a_out / (n_out + N - a_out)))
                scored.append((delta / math.sqrt(1 / a_in + 1 / a_out), w))
            scored.sort(reverse=True)
            # min_z matters most for the rare classes: at 184 Geo-political rows a
            # plain top-k fills with function words (ada, avar, beda) that fire
            # everywhere and turn the tag into noise. The threshold lets a small
            # class contribute five clean terms instead of forty dirty ones.
            self.gaz[cname] = {w for z, w in scored[:self.top_k] if z >= self.min_z}
        return self

    def tags(self, text):
        toks = _types(text)
        out = []
        for cname in self.label_names:
            if toks & self.gaz.get(cname, ()):
                out.append(TAG_TOPIC.get(cname, f"{cname.lower()} topic"))
        if NECESSITATIVE.search(text.lower()):
            out.append("demand mood")
        if toks & SECOND_PERSON:
            out.append("second person address")
        if toks & RESPECTFUL:
            out.append("respectful address")
        return out

    def transform(self, texts):
        """text -> "text | tag ; tag". Returns a plain list of strings."""
        out = []
        for t in texts:
            tg = self.tags(t)
            out.append(f"{t} | {' ; '.join(tg)}" if tg else t)
        return out


def describe(lex, n=8):
    return "\n".join(f"    {c:14s} " + " ".join(sorted(lex.gaz[c])[:n])
                     for c in lex.label_names)
