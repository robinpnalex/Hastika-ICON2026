"""Train a small offensiveness tagger, then tag every Task B comment with it.

STANDING IN FOR SENTIMENT, AND WHY
----------------------------------
There is no usable romanized-Kannada sentiment corpus. The DravidianCodeMix set
on the hub is Tamil, and nothing else in the same script convention is public. So
this trains a binary Hate / Non-Hate tagger on data/external/offenseval_kn.csv --
the same file TAPT already reads -- and appends its verdict to each comment.

A RULES NOTE. TAPT uses that file's TEXT only; this uses its LABELS. That is a
different call from the one already made, so decide before submitting this arm.

Output is a JSON {cleaned comment: tag} consumed by
`train.py --text-transform polarity --polarity-map ...`. Written that way so the
tagger runs once and all five classifier seeds reuse it.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from hastika.common.paths import RAW_DATA_DIR                    # noqa: E402
from hastika.common.preprocessing import clean                   # noqa: E402
from hastika.models.muril import Comments, MurilClassifier, make_collate, predict  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
TAGS = ["neutral polarity", "offensive polarity"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/muril-base-cased")
    ap.add_argument("--external", default="data/external/offenseval_kn.csv")
    ap.add_argument("--out", default="artifacts/runs/polarity_map.json")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)

    ext = pd.read_csv(ROOT / a.external)
    X = ext["Comment"].map(lambda t: clean(t, demojize=True)).values
    y = (ext["Label"] == "Hate").astype(int).values
    print(f"tagger trains on {len(y)} external rows, {y.sum()} offensive", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = MurilClassifier(a.model, "meanmax", 0.1, 0, n_classes=2).to(device)
    coll = make_collate(tok, a.max_len)
    dl = DataLoader(Comments(X, y), batch_size=a.bs, shuffle=True, collate_fn=coll,
                    drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, int(0.06 * len(dl) * a.epochs),
                                            len(dl) * a.epochs)
    w = torch.tensor(len(y) / (2 * np.maximum(np.bincount(y, minlength=2), 1)),
                     dtype=torch.float, device=device)
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        for b in dl:
            b = {k: v.to(device) for k, v in b.items()}
            labels = b.pop("labels")
            loss = F.cross_entropy(model(**b), labels, weight=w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            tot += loss.item()
        print(f"  epoch {ep}/{a.epochs} loss {tot/len(dl):.4f}", flush=True)

    # tag every Task B comment, train and test, in the cleaned form train.py uses
    texts = []
    for f, col in [("multiclass_train.csv", "Comment"),
                   ("multiclass_validation_inputs.csv", "Comment")]:
        texts += [clean(t, demojize=True) for t in pd.read_csv(RAW_DATA_DIR / f)[col]]
    texts = sorted(set(texts))
    dl = DataLoader(Comments(texts), batch_size=64, collate_fn=coll)
    p = predict(model, dl, device, None).argmax(1)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({t: TAGS[i] for t, i in zip(texts, p)}, ensure_ascii=False))
    print(f"wrote {out}: {len(texts)} comments, "
          f"{int((p == 1).sum())} tagged offensive", flush=True)


if __name__ == "__main__":
    main()
