"""Task-adaptive MLM pretraining for Task B, on in-domain text only.

MuRIL saw transliterated Indic during pretraining, but not this register: it
splits these comments into 2.28 wordpieces per whitespace word, because
romanized Kannada social text is agglutinative and spelled ad hoc. A short
masked-LM pass over the task's own domain, before any classifier exists, is the
standard fix (Gururangan et al., "Don't Stop Pretraining", ACL 2020) and costs
one pass over ~6.4k comments.

WHAT GOES IN THE CORPUS, AND WHAT MUST NOT
------------------------------------------
Only multiclass_train.csv and data/external/offenseval_kn.csv. No labels are
read from either -- masked-LM needs text alone -- but the file list still
matters, because two tempting sources would poison the evaluation:

  * binary_train.csv. 319 of the 395 ids in multiclass_validation_inputs.csv
    also appear in binary_train.csv, so Task A's training file contains 81% of
    Task B's test comments. Task A and Task B are separate problems sharing a
    comment pool; pulling in Task A text means fitting the language model to the
    text you are about to be scored on.
  * multiclass_validation_inputs.csv, for the same reason, directly.

Both are refused below rather than left to a flag, which is why --extra exists
but will not accept them.
"""
import argparse
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, get_cosine_schedule_with_warmup)

from prep import clean

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ["data/multiclass_train.csv", "data/external/offenseval_kn.csv"]
FORBIDDEN = {"binary_train.csv", "binary_validation_inputs.csv",
             "multiclass_validation_inputs.csv"}


class Lines(torch.utils.data.Dataset):
    def __init__(self, enc):
        self.enc = enc

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}


def load_corpus(paths, demoji):
    texts = []
    for p in paths:
        name = pathlib.Path(p).name
        if name in FORBIDDEN:
            sys.exit(f"refusing {name}: see this module's docstring -- it overlaps the "
                     f"Task B test inputs")
        df = pd.read_csv(ROOT / p if not pathlib.Path(p).is_absolute() else p)
        col = "Comment" if "Comment" in df.columns else df.columns[-1]
        texts += [clean(t, demojize=demoji) for t in df[col].astype(str)]
        print(f"  {p}: {len(df)} rows", flush=True)
    texts = [t for t in texts if len(t.split()) >= 2]
    seen, out = set(), []
    for t in texts:                       # a repeated comment is not extra evidence
        if t.casefold() not in seen:
            seen.add(t.casefold())
            out.append(t)
    print(f"corpus: {len(out)} unique comments (from {len(texts)})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/muril-base-cased",
                    help="a checkpoint with an MLM head. The abusive-tuned MuRIL is a "
                         "sequence-classification checkpoint, so pointing here at it "
                         "trains a fresh MLM head over its encoder -- allowed, but use a "
                         "lower --lr if you do")
    ap.add_argument("--out", default="work/runs/tapt-muril")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="further in-domain CSVs; the Task A and test files are refused")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=4,
                    help="micro-batch. MuRIL's MLM head emits a bs x seq x 197285 "
                         "logits tensor and cross-entropy upcasts it to fp32, so bs=16 "
                         "at --max-len 192 asks for 2.26 GiB in a single allocation and "
                         "a 16 GB T4 cannot hold it. Raise --grad-accum instead: the "
                         "effective batch is --bs x --grad-accum")
    ap.add_argument("--grad-accum", type=int, default=4,
                    help="micro-batches per optimizer step")
    ap.add_argument("--eval-bs", type=int, default=8,
                    help="perplexity runs under no_grad but allocates the same logits")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--mlm-prob", type=float, default=0.15)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--val-frac", type=float, default=0.05,
                    help="held out to report MLM perplexity, which is the only honest "
                         "signal that this pass did anything")
    ap.add_argument("--no-demojize", action="store_true")
    ap.add_argument("--amp", choices=["auto", "off", "fp16", "bf16"], default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0,
                    help="use only the first N comments -- smoke-testing a new machine")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} model={args.model}", flush=True)
    texts = load_corpus(DEFAULT_CORPUS + args.extra, not args.no_demojize)
    if args.limit:
        texts = texts[:args.limit]
        print(f"LIMIT: {len(texts)} comments (smoke test)", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model).to(device)
    enc = tok(texts, truncation=True, max_length=args.max_len, padding=False)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(texts))
    n_val = int(len(texts) * args.val_frac)
    split = {"val": idx[:n_val], "train": idx[n_val:]}
    sets = {k: Lines({f: [enc[f][i] for i in v] for f in enc}) for k, v in split.items()}

    coll = DataCollatorForLanguageModeling(tok, mlm_probability=args.mlm_prob)
    tr = DataLoader(sets["train"], batch_size=args.bs, shuffle=True, collate_fn=coll)
    va = DataLoader(sets["val"], batch_size=args.eval_bs, collate_fn=coll)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, eps=1e-6)
    accum = max(1, args.grad_accum)
    steps = max(1, math.ceil(len(tr) / accum) * args.epochs)
    sched = get_cosine_schedule_with_warmup(opt, int(args.warmup * steps), steps)
    dtype = (None if device.type != "cuda" or args.amp == "off" else
             torch.bfloat16 if args.amp == "bf16" or
             (args.amp == "auto" and torch.cuda.is_bf16_supported()) else torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    @torch.no_grad()
    def perplexity():
        model.eval()
        tot, n = 0.0, 0
        for b in va:
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                loss = model(**b).loss
            tot += loss.item() * len(b["input_ids"])
            n += len(b["input_ids"])
        return math.exp(tot / max(n, 1))

    print(f"effective batch {args.bs} x {accum} = {args.bs * accum}, "
          f"{math.ceil(len(tr) / accum)} optimizer steps/epoch", flush=True)
    print(f"held-out perplexity before {perplexity():.2f}", flush=True)
    opt.zero_grad(set_to_none=True)
    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for step, b in enumerate(tr, 1):
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                loss = model(**b).loss
            run_loss += loss.item()
            # scale by accum so the gradient matches one batch of bs*accum rows
            scaler.scale(loss / accum).backward()
            if step % accum and step != len(tr):
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
        print(f"  epoch {ep}/{args.epochs} train loss {run_loss/len(tr):.4f} "
              f"held-out perplexity {perplexity():.2f}", flush=True)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"\nwrote {out}\nuse it with:  python work/muril_b.py --model {args.out} --tag muril_b_tapt",
          flush=True)


if __name__ == "__main__":
    main()
