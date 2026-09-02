"""Best-practice MuRIL fine-tune for HASTIKA Task A (binary hate speech).

Every non-default choice here is justified by a measurement on this dataset --
see work/SETUP.md. Summary of what drove the design:

  * MuRIL's wordpiece vocab has no emoji, and BERT wordpiece maps the WHOLE
    whitespace word containing one to [UNK] ("tintara<emoji>" -> [UNK]).
    Measured 0.80% -> 0.05% UNK with --demojize, so demojize defaults to ON here
    (unlike train_xlmr.py, where sentencepiece keeps emoji natively).
  * XLM-R needed two full epochs on this data just to escape the trivial
    solution (loss pinned at ln 2). Hence 6 epochs, 6% warmup, and evaluation
    twice per epoch so best-checkpoint selection is not blind between epochs.
  * 6,446 training rows is small and BERT fine-tuning is high-variance at that
    size, so: layer-wise LR decay, top-layer re-init, FGM adversarial training,
    EMA, and multi-seed averaging -- all aimed at variance, not capacity.
  * Labels are near-balanced (3286 Non-Hate / 3160 Hate), so no class weighting.

Outputs are written in the same layout train_xlmr.py uses, so ensemble.py can
blend these runs with the SVM and the other encoders.
"""
import argparse
import copy
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from prep import clean

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "work" / "runs"
SPLIT_SEED = 42          # must match baseline_svm.py so OOF rows line up
N_SPLITS = 5


# ---------------------------------------------------------------- data

class Comments(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = list(texts)
        self.labels = None if labels is None else list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], (-1 if self.labels is None else self.labels[i])


def make_collate(tok, max_len):
    def collate(batch):
        texts, labels = zip(*batch)
        enc = tok(list(texts), truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc
    return collate


# ---------------------------------------------------------------- model

class MurilClassifier(nn.Module):
    """MuRIL encoder + a pooling head.

    `meanmax` (concat of masked mean and masked max over token states) is the
    default rather than [CLS]: these comments are short (p50 18 tokens) and the
    hate signal is usually one or two slur tokens, which max-pooling picks up and
    [CLS] tends to wash out.
    """

    def __init__(self, name, pooling="meanmax", dropout=0.1, reinit_layers=0, n_classes=2):
        super().__init__()
        cfg = AutoConfig.from_pretrained(name)
        cfg.hidden_dropout_prob = dropout
        cfg.attention_probs_dropout_prob = dropout
        self.backbone = AutoModel.from_pretrained(name, config=cfg)
        self.pooling = pooling
        h = cfg.hidden_size
        width = {"cls": h, "mean": h, "meanmax": 2 * h, "last4": 4 * h}[pooling]
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width, n_classes)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        if reinit_layers:
            self._reinit_top(reinit_layers)

    def _reinit_top(self, n):
        """Re-initialize the top n encoder layers.

        The topmost layers are the most MLM-specialized and transfer worst; on
        small datasets re-initializing them is a consistent, cheap win
        (Zhang et al., 'Revisiting Few-sample BERT Fine-tuning').
        """
        for layer in self.backbone.encoder.layer[-n:]:
            for m in layer.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_(mean=0.0, std=self.backbone.config.initializer_range)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.LayerNorm):
                    m.weight.data.fill_(1.0)
                    m.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids=None, **_):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        out = self.backbone(**kw, output_hidden_states=(self.pooling == "last4"))
        hs = out.last_hidden_state
        m = attention_mask.unsqueeze(-1).float()

        if self.pooling == "cls":
            pooled = hs[:, 0]
        elif self.pooling == "mean":
            pooled = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
        elif self.pooling == "meanmax":
            mean = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
            mx = hs.masked_fill(m == 0, -1e4).max(1).values
            pooled = torch.cat([mean, mx], dim=-1)
        else:  # last4: concat [CLS] of the top four layers
            pooled = torch.cat([h[:, 0] for h in out.hidden_states[-4:]], dim=-1)

        return self.head(self.dropout(pooled))


# ------------------------------------------------- training utilities

class FGM:
    """Fast Gradient Method adversarial training on the word embeddings.

    One extra forward/backward per step (~1.8x cost). Reliable ~0.5-1 point on
    small, noisy text datasets: it forces the classifier to be locally flat
    around each embedding, which matters here because 70% of vocabulary types are
    hapax and their embeddings are otherwise free to overfit.
    """

    def __init__(self, model, eps=1.0):
        self.model, self.eps, self.backup = model, eps, {}

    def _emb(self):
        return self.model.backbone.embeddings.word_embeddings

    def attack(self):
        p = self._emb().weight
        if p.grad is None:
            return False
        self.backup["w"] = p.data.clone()
        norm = torch.norm(p.grad)
        if torch.isfinite(norm) and norm != 0:
            p.data.add_(self.eps * p.grad / norm)
            return True
        return False

    def restore(self):
        if "w" in self.backup:
            self._emb().weight.data = self.backup.pop("w")


class EMA:
    """Exponential moving average of weights; evaluated instead of the raw model."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def state_dict_for_eval(self, model):
        sd = copy.deepcopy(model.state_dict())
        sd.update(self.shadow)
        return sd


def llrd_param_groups(model, lr, head_lr, decay=0.9, weight_decay=0.01):
    """Layer-wise learning-rate decay.

    Lower layers encode general lexical/morphological structure that we want to
    preserve -- especially here, where MuRIL's value is its pretrained handling
    of transliterated Indic. Upper layers adapt fastest. Each layer down from the
    top gets lr * decay**k.
    """
    no_decay = ("bias", "LayerNorm.weight")
    layers = [model.backbone.embeddings] + list(model.backbone.encoder.layer)
    groups, n = [], len(layers)
    for i, layer in enumerate(layers):
        layer_lr = lr * (decay ** (n - 1 - i))
        for nd in (True, False):
            params = [p for k, p in layer.named_parameters()
                      if p.requires_grad and (any(x in k for x in no_decay) == nd)]
            if params:
                groups.append({"params": params, "lr": layer_lr,
                               "weight_decay": 0.0 if nd else weight_decay})
    head = [p for k, p in model.named_parameters()
            if p.requires_grad and not k.startswith("backbone.")]
    groups.append({"params": head, "lr": head_lr, "weight_decay": weight_decay})
    return groups


def amp_dtype(device, mode):
    if device.type != "cuda" or mode == "off":
        return None
    if mode == "bf16" or (mode == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return torch.float16


@torch.no_grad()
def predict(model, loader, device, dtype):
    model.eval()
    out = []
    for batch in loader:
        batch.pop("labels", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            logits = model(**batch)
        out.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def best_threshold(y, p1, lo=0.30, hi=0.70):
    """Macro-F1-optimal decision threshold, searched on out-of-fold scores.

    Restricted to [lo, hi]: the classes are near-balanced, so the honest optimum
    is close to 0.5. An extreme threshold means the model is degenerate (a smoke
    run with too few steps will happily "improve" by predicting one class at
    t=0.94) and carrying that over to the test set would be actively harmful.
    """
    ts = np.unique(np.round(p1, 3))
    ts = ts[(ts >= lo) & (ts <= hi)]
    best = (0.5, f1_score(y, (p1 > 0.5).astype(int), average="macro"))
    for t in ts:
        f = f1_score(y, (p1 > t).astype(int), average="macro")
        if f > best[1]:
            best = (float(t), f)
    return best


# ---------------------------------------------------------------- train

def train_fold(args, tok, X_tr, y_tr, X_va, y_va, X_test, device, tag):
    torch.manual_seed(args.seed)
    model = MurilClassifier(args.model, args.pooling, args.dropout, args.reinit_layers).to(device)
    collate = make_collate(tok, args.max_len)
    tr = DataLoader(Comments(X_tr, y_tr), batch_size=args.bs, shuffle=True,
                    collate_fn=collate, drop_last=True)
    va = DataLoader(Comments(X_va, y_va), batch_size=args.eval_bs, collate_fn=collate)
    te = DataLoader(Comments(X_test), batch_size=args.eval_bs, collate_fn=collate)

    opt = torch.optim.AdamW(
        llrd_param_groups(model, args.lr, args.head_lr, args.llrd, args.weight_decay), eps=1e-6)
    steps = max(1, (len(tr) // args.grad_accum) * args.epochs)
    sched = get_cosine_schedule_with_warmup(opt, int(args.warmup * steps), steps)
    dtype = amp_dtype(device, args.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    fgm = FGM(model, args.fgm_eps) if args.fgm else None
    ema = EMA(model, args.ema_decay) if args.ema else None

    # evaluate this many times per epoch -- epoch-level granularity is too coarse
    # given how sharply val F1 moves between epochs on this dataset
    eval_every = max(1, len(tr) // args.evals_per_epoch)
    best = {"f1": -1.0, "va": None, "te": None}
    step_i = 0

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        for step, batch in enumerate(tr, 1):
            model.train()
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                logits = model(**batch)
                loss = F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)
                if args.rdrop:
                    logits2 = model(**batch)
                    loss = 0.5 * (loss + F.cross_entropy(logits2, labels,
                                                         label_smoothing=args.label_smoothing))
                    kl = 0.5 * (F.kl_div(F.log_softmax(logits, -1), F.softmax(logits2, -1),
                                         reduction="batchmean")
                                + F.kl_div(F.log_softmax(logits2, -1), F.softmax(logits, -1),
                                           reduction="batchmean"))
                    loss = loss + args.rdrop * kl
            scaler.scale(loss / args.grad_accum).backward()

            if fgm is not None and fgm.attack():
                with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                    adv = F.cross_entropy(model(**batch), labels,
                                          label_smoothing=args.label_smoothing)
                scaler.scale(adv / args.grad_accum).backward()
                fgm.restore()

            if step % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
                step_i += 1

            if step % eval_every == 0 or step == len(tr):
                if ema is not None:
                    raw = copy.deepcopy(model.state_dict())
                    model.load_state_dict(ema.state_dict_for_eval(model))
                p_va = predict(model, va, device, dtype)
                f1 = f1_score(y_va, p_va.argmax(1), average="macro")
                if f1 > best["f1"]:
                    best = {"f1": f1, "va": p_va, "te": predict(model, te, device, dtype)}
                    mark = " *"
                else:
                    mark = ""
                if ema is not None:
                    model.load_state_dict(raw)
                print(f"  [{tag}] ep{ep} {step}/{len(tr)} loss {loss.item():.4f} "
                      f"val {f1:.4f}{mark} ({(time.time()-t0)/step:.2f}s/step)", flush=True)

        print(f"  [{tag}] ep{ep} done {(time.time()-t0)/60:.1f}min  best {best['f1']:.4f}",
              flush=True)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best["f1"], best["va"], best["te"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/muril-base-cased")
    ap.add_argument("--tag", default="muril")
    # data
    ap.add_argument("--max-len", type=int, default=128, help="p99 is 121 tokens")
    ap.add_argument("--no-demojize", action="store_true",
                    help="disable emoji->:name: rewriting (MuRIL needs it; see module docstring)")
    # schedule
    ap.add_argument("--folds", type=int, default=N_SPLITS, help="0 = 15%% holdout")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="average several seeds; BERT fine-tuning is high-variance at this size")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--eval-bs", type=int, default=64)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--evals-per-epoch", type=int, default=2)
    # optimization
    ap.add_argument("--lr", type=float, default=3e-5, help="top encoder layer LR")
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--llrd", type=float, default=0.9, help="layer-wise LR decay factor")
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    # architecture / regularization
    ap.add_argument("--pooling", choices=["meanmax", "mean", "cls", "last4"], default="meanmax")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--reinit-layers", type=int, default=2)
    ap.add_argument("--fgm", action="store_true", default=True)
    ap.add_argument("--no-fgm", dest="fgm", action="store_false")
    ap.add_argument("--fgm-eps", type=float, default=1.0)
    ap.add_argument("--ema", action="store_true", default=True)
    ap.add_argument("--no-ema", dest="ema", action="store_false")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--rdrop", type=float, default=0.0,
                    help="R-Drop KL weight; 0 disables. Try 0.3-1.0. Doubles cost, "
                         "and stacks with --fgm at ~2.8x baseline")
    ap.add_argument("--amp", choices=["auto", "off", "fp16", "bf16"], default="auto")
    ap.add_argument("--seed", type=int, default=42)   # set per-seed in the loop
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="train on only the first N rows -- smoke-testing a new machine")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "-"
    print(f"device={device} ({gpu}) model={args.model} pooling={args.pooling} "
          f"fgm={args.fgm} ema={args.ema} rdrop={args.rdrop} reinit={args.reinit_layers} "
          f"seeds={args.seeds}", flush=True)

    train = pd.read_csv(ROOT / "data" / "binary_train.csv")
    test = pd.read_csv(ROOT / "data" / "binary_validation_inputs.csv")
    demoji = not args.no_demojize
    X = train["Comment"].map(lambda t: clean(t, demojize=demoji)).values
    y = (train["Label"] == "Hate").astype(int).values
    X_test = test["Comment"].map(lambda t: clean(t, demojize=demoji)).values

    if args.limit:
        X, y = X[:args.limit], y[:args.limit]
        print(f"LIMIT: using {len(y)} training rows (smoke test)", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    run = RUNS / args.tag
    run.mkdir(parents=True, exist_ok=True)

    oof = np.zeros((len(y), 2))
    test_probs = np.zeros((len(X_test), 2))
    n_seeds = len(args.seeds)

    for seed in args.seeds:
        args.seed = seed
        if args.folds and args.folds > 1:
            skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SPLIT_SEED)
            for k, (tr_i, va_i) in enumerate(skf.split(X, y), 1):
                print(f"===== seed {seed} fold {k}/{args.folds} =====", flush=True)
                f1, p_va, p_te = train_fold(args, tok, X[tr_i], y[tr_i], X[va_i], y[va_i],
                                            X_test, device, f"s{seed}f{k}")
                oof[va_i] += p_va / n_seeds
                test_probs += p_te / (args.folds * n_seeds)
        else:
            from sklearn.model_selection import train_test_split
            tr_i, va_i = train_test_split(np.arange(len(y)), test_size=0.15,
                                          stratify=y, random_state=SPLIT_SEED)
            f1, p_va, p_te = train_fold(args, tok, X[tr_i], y[tr_i], X[va_i], y[va_i],
                                        X_test, device, f"s{seed}holdout")
            oof[va_i] += p_va / n_seeds
            test_probs += p_te / n_seeds

    if args.folds and args.folds > 1:
        pred = oof.argmax(1)
        print(f"\nOOF macro-F1 {f1_score(y, pred, average='macro'):.4f} "
              f"acc {accuracy_score(y, pred):.4f}")
        t, tf1 = best_threshold(y, oof[:, 1])
        print(f"threshold-tuned OOF macro-F1 {tf1:.4f} at p(Hate) > {t:.3f}")
        np.save(run / "oof_probs.npy", oof)
        # Use the tuned threshold only if it is a real gain, not noise.
        thr = t if tf1 - f1_score(y, pred, average="macro") > 0.002 else 0.5
    else:
        thr = 0.5
        np.save(run / "holdout_probs.npy", oof)

    np.save(run / "test_probs.npy", test_probs)
    pd.DataFrame({"id": test["id"],
                  "label": np.where(test_probs[:, 1] > thr, "Hate", "Non-Hate")}
                 ).to_csv(run / "predictions.csv", index=False)
    print(f"wrote {run}/predictions.csv (threshold {thr:.3f})")


if __name__ == "__main__":
    main()
