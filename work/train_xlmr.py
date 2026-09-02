"""Fine-tune XLM-RoBERTa for HASTIKA Task A (binary hate speech).

Plain PyTorch loop -- no Trainer, so it behaves the same on transformers 4.x and 5.x.
Runs a stratified holdout by default; --folds N does full CV and averages the
fold models' probabilities into the submission.
"""
import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from prep import clean

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "work" / "runs"


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
        # pad to the longest item in the batch, not to max_len -- on CPU this is
        # roughly a 2x speedup, since p95 length is ~68 tokens.
        enc = tok(list(texts), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc

    return collate


@torch.no_grad()
def predict(model, loader, device, dtype=None):
    model.eval()
    probs = []
    for batch in loader:
        batch.pop("labels", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            logits = model(**batch).logits
        probs.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(probs)


def amp_ctx(args, device):
    """Returns (autocast_dtype or None, use_scaler)."""
    if device.type != "cuda" or args.amp == "off":
        return None, False
    if args.amp == "bf16" or (args.amp == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16, False          # bf16 needs no loss scaling
    return torch.float16, True


def train_one(args, tok, X_tr, y_tr, X_va, y_va, X_test, device, tag):
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
    collate = make_collate(tok, args.max_len)
    tr = DataLoader(Comments(X_tr, y_tr), batch_size=args.bs, shuffle=True, collate_fn=collate)
    va = DataLoader(Comments(X_va, y_va), batch_size=args.eval_bs, collate_fn=collate)
    te = DataLoader(Comments(X_test), batch_size=args.eval_bs, collate_fn=collate)

    no_decay = ("bias", "LayerNorm.weight")
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(grouped, lr=args.lr)
    dtype, use_scaler = amp_ctx(args, device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    total = (len(tr) // args.grad_accum) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

    best_f1, best_va, best_te = -1.0, None, None
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for step, batch in enumerate(tr, 1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                loss = model(**batch).loss / args.grad_accum
            scaler.scale(loss).backward()
            if step % args.grad_accum == 0 or step == len(tr):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            running += loss.item() * args.grad_accum
            if step % args.log_every == 0:
                print(f"  [{tag}] ep{ep} {step}/{len(tr)} loss {running/step:.4f} "
                      f"({(time.time()-t0)/step:.2f}s/step)", flush=True)

        p_va = predict(model, va, device, dtype)
        f1 = f1_score(y_va, p_va.argmax(1), average="macro")
        print(f"  [{tag}] ep{ep} done in {(time.time()-t0)/60:.1f}min -- "
              f"val macro-F1 {f1:.4f} acc {accuracy_score(y_va, p_va.argmax(1)):.4f}", flush=True)
        if f1 > best_f1:
            best_f1, best_va = f1, p_va
            best_te = predict(model, te, device, dtype)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_f1, best_va, best_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--folds", type=int, default=0, help="0 = single stratified holdout")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--eval-bs", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=0, help="0 = torch default")
    ap.add_argument("--demojize", action="store_true",
                    help="rewrite emoji as :names: -- required for MuRIL, whose wordpiece "
                         "vocab has no emoji and drops the whole surrounding word to [UNK]")
    ap.add_argument("--amp", choices=["auto", "off", "fp16", "bf16"], default="auto",
                    help="mixed precision; 'auto' picks bf16 on Ampere+, fp16 on older GPUs, off on CPU")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--tag", default="xlmr")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "-"
    print(f"device={device} ({gpu}) threads={torch.get_num_threads()} "
          f"model={args.model} amp={args.amp} demojize={args.demojize}", flush=True)

    train = pd.read_csv(ROOT / "data" / "binary_train.csv")
    test = pd.read_csv(ROOT / "data" / "binary_validation_inputs.csv")
    prep_fn = lambda t: clean(t, demojize=args.demojize)
    X = train["Comment"].map(prep_fn).values
    y = (train["Label"] == "Hate").astype(int).values
    X_test = test["Comment"].map(prep_fn).values

    tok = AutoTokenizer.from_pretrained(args.model)
    OUT.mkdir(parents=True, exist_ok=True)
    run = OUT / args.tag
    run.mkdir(exist_ok=True)

    if args.folds and args.folds > 1:
        oof = np.zeros((len(y), 2))
        test_probs = np.zeros((len(X_test), 2))
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        for k, (tr_i, va_i) in enumerate(skf.split(X, y), 1):
            print(f"===== fold {k}/{args.folds} =====", flush=True)
            f1, p_va, p_te = train_one(args, tok, X[tr_i], y[tr_i], X[va_i], y[va_i],
                                       X_test, device, f"f{k}")
            oof[va_i] = p_va
            test_probs += p_te / args.folds
        pred = oof.argmax(1)
        print(f"\nOOF macro-F1 {f1_score(y, pred, average='macro'):.4f} "
              f"acc {accuracy_score(y, pred):.4f}")
        np.save(run / "oof_probs.npy", oof)
    else:
        tr_i, va_i = train_test_split(np.arange(len(y)), test_size=0.15,
                                      stratify=y, random_state=args.seed)
        f1, p_va, test_probs = train_one(args, tok, X[tr_i], y[tr_i], X[va_i], y[va_i],
                                         X_test, device, "holdout")
        print(f"\nbest holdout macro-F1 {f1:.4f}")
        np.save(run / "holdout_probs.npy", p_va)
        np.save(run / "holdout_idx.npy", va_i)

    np.save(run / "test_probs.npy", test_probs)
    sub = pd.DataFrame({
        "id": test["id"],
        "label": np.where(test_probs.argmax(1) == 1, "Hate", "Non-Hate"),
    })
    sub.to_csv(run / "predictions.csv", index=False)
    print(f"wrote {run/'predictions.csv'}  ({sub['label'].value_counts().to_dict()})")


if __name__ == "__main__":
    main()
