"""The two axes this label set actually encodes, as regex lexicons.

MEASURED STRUCTURE
------------------
The six Task B labels are not one taxonomy. Gender, Political, Religion and
Geo-political name *who is targeted*. Violence names *what is threatened*.
Others is the residual of the first axis. Measured on multiclass_train.csv,
when a violent verb appears alongside a target marker the target wins:

    target marker present   n    -> Gender  -> Political  -> Religion  -> Violence
    feminine               87        53%          9%          11%          16%
    political              58         3%         66%          14%           9%
    religion               52         0%         12%          69%          13%
    none                  242        33%          7%           9%          31%

So the operating rule is: Violence means a threat with no identifiable target
group. That is a conjunction with a negation -- act present, every target
absent -- which is a shape a linear model learns badly from 221 examples.

WHAT THESE ARE FOR
------------------
`act()` is weak supervision for an auxiliary training objective, not a
classifier. As a Violence classifier it is poor: precision 0.24, recall 0.48.
But as a detector of *violent language present*, which is what the auxiliary
head is asked to predict, it fires on 437 rows rather than the 221 the Violence
class provides, so the encoder gets twice the signal for the concept.

Adding these as input features was measured and does not work: on the TF-IDF
floor it moved macro-F1 from 0.5948 to 0.5915. Character n-grams already see
these words. The auxiliary head is a different mechanism -- it shapes the
representation instead of appending nine numbers to 200k sparse features.
"""
import re

ACT = re.compile(r"\b(hodi|hodey|hode|odi|haak|haku|hak|benki|kollu|kol|sayis|say|"
                 r"kadi|chappali|chapli|bomb|rape|murder|kill|gallige|suttu|kachis|"
                 r"muri|nashp|hosak|gun|weapon|thivi|jail|gun)\w*", re.I)

TARGETS = {
    "feminine":  re.compile(r"\b(avl|ivl|avalu|ivalu|hennu|hudgi|magalu|maglu|sule|"
                            r"sulle|dagar|randi|mindri|batte|akka|amma)\w*", re.I),
    "political": re.compile(r"\b(bjp|congress|congi|modi|siddu|siddaram|bommai|speaker|"
                            r"mla|cm|party|govt|minister|election|sarkara)\b", re.I),
    "religion":  re.compile(r"\b(muslim|muslims|islam|hindu|hindus|hijab|allah|saab|"
                            r"mulla|namaz|mamsa|mutton|jati|dharma|bajrangi)\b", re.I),
    "geo":       re.compile(r"\b(pakistan|tamil|telugu|andhra|kannada|kashmir|madrasi|"
                            r"hindi|kerala|bengali|desha|rajya)\b", re.I),
    "media":     re.compile(r"\b(btv|tv|news|channel|media|interview|anchor|bucket|"
                            r"youtube|video)\b", re.I),
}


def act(text):
    """1 if the comment contains violent language. Weak, plentiful supervision."""
    return int(bool(ACT.search(text)))


def targets(text):
    """Which target lexicons fire. Used for analysis, not fed to the model."""
    return [k for k, rx in TARGETS.items() if rx.search(text)]


def aux_labels(texts):
    return [act(t) for t in texts]
