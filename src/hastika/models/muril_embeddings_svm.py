"""Frozen MuRIL embeddings followed by an SVM classifier for Task A.

The experiment extracts masked mean+max MuRIL embeddings from cleaned, demojized
comments, scores an RBF SVM on the repository's fixed 85/15 holdout, then refits
the SVM on every deduplicated labelled row for validation predictions. The final
fit is deliberately separate from the holdout measurement.
"""
import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from hastika.common.paths import RAW_DATA_DIR, RUNS_DIR
from hastika.common.preprocessing import clean, dedupe_index

SPLIT_SEED = 42


class Texts(Dataset):
    def __init__(self, texts):
        self.texts = list(texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i]


def make_collate(tokenizer, max_len):
    def collate(batch):
        return tokenizer(list(batch), truncation=True, max_length=max_len,
                         padding=True, return_tensors="pt")
    return collate


def meanmax(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).bool()
    masked = last_hidden.masked_fill(~mask, -torch.inf)
    mean = (last_hidden * mask).sum(1) / attention_mask.sum(1, keepdim=True).clamp_min(1)
    maximum = masked.max(1).values
    return torch.cat([mean, maximum], dim=-1)


def encode(texts, model_name, batch_size, max_len, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    loader = DataLoader(Texts(texts), batch_size=batch_size, shuffle=False,
                        collate_fn=make_collate(tokenizer, max_len),
                        pin_memory=device.type == "cuda")
    chunks = []
    amp_dtype = torch.float16 if device.type == "cuda" else None
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                hidden = model(**batch).last_hidden_state
            chunks.append(meanmax(hidden.float(), batch["attention_mask"]).cpu().numpy())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(chunks).astype("float32")


def build_svm(kernel, C, gamma, cache_size):
    return SVC(kernel=kernel, C=C, gamma=gamma, class_weight="balanced",
               cache_size=cache_size)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/muril-base-cased")
    ap.add_argument("--tag", default="task_a_muril_embeddings_svm")
    ap.add_argument("--kernel", choices=["rbf", "linear"], default="rbf")
    ap.add_argument("--C", type=float, default=2.0)
    ap.add_argument("--gamma", default="scale")
    ap.add_argument("--cache-size", type=float, default=4096)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--valid-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--no-demojize", action="store_true")
    ap.add_argument("--full-fit", action="store_true",
                    help="skip holdout scoring and fit the SVM on every row")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} model={args.model} kernel={args.kernel} C={args.C} "
          f"pooling=meanmax demojize={not args.no_demojize}", flush=True)

    train = pd.read_csv(RAW_DATA_DIR / "binary_train.csv")
    test = pd.read_csv(RAW_DATA_DIR / "binary_validation_inputs.csv")
    keep = dedupe_index(train["Comment"].tolist(), train["Label"].tolist(), "task A")
    train = train.iloc[keep].reset_index(drop=True)
    X = train["Comment"].map(lambda x: clean(x, demojize=not args.no_demojize)).tolist()
    y = (train["Label"] == "Hate").astype(int).to_numpy()
    X_test = test["Comment"].map(lambda x: clean(x, demojize=not args.no_demojize)).tolist()
    print(f"deduplicated labelled rows: {len(train)}; validation inputs: {len(test)}", flush=True)

    run = RUNS_DIR / args.tag
    run.mkdir(parents=True, exist_ok=True)
    train_emb = encode(X, args.model, args.batch_size, args.max_len, device)
    test_emb = encode(X_test, args.model, args.batch_size, args.max_len, device)
    np.save(run / "train_embeddings.npy", train_emb)
    np.save(run / "test_embeddings.npy", test_emb)
    print(f"embeddings: train={train_emb.shape} validation={test_emb.shape}", flush=True)

    threshold = 0.0
    if not args.full_fit:
        tr_i, va_i = train_test_split(np.arange(len(y)), test_size=args.valid_size,
                                       stratify=y, random_state=args.seed)
        clf = build_svm(args.kernel, args.C, args.gamma, args.cache_size)
        clf.fit(train_emb[tr_i], y[tr_i])
        decision = clf.decision_function(train_emb[va_i])
        pred = (decision >= threshold).astype(int)
        print(f"holdout macro-F1 {f1_score(y[va_i], pred, average='macro'):.4f} "
              f"acc {accuracy_score(y[va_i], pred):.4f}", flush=True)
        print(classification_report(y[va_i], pred,
                                    target_names=["Non-Hate", "Hate"], digits=3), flush=True)
        np.save(run / "holdout_idx.npy", va_i)
        np.save(run / "holdout_decision.npy", decision)

    clf = build_svm(args.kernel, args.C, args.gamma, args.cache_size)
    clf.fit(train_emb, y)
    test_decision = clf.decision_function(test_emb)
    labels = np.where(test_decision >= threshold, "Hate", "Non-Hate")
    np.save(run / "test_decision.npy", test_decision)
    pd.DataFrame({"id": test["id"], "label": labels}).to_csv(
        run / "predictions.csv", index=False)
    with open(run / "config.json", "w") as f:
        json.dump({"model": args.model, "pooling": "meanmax", "demojize": not args.no_demojize,
                   "kernel": args.kernel, "C": args.C, "gamma": args.gamma,
                   "training_rows": len(train), "full_fit": True,
                   "threshold": threshold}, f, indent=2)
    print(f"full-data SVM fit on {len(train)} rows")
    print(f"wrote {run}/predictions.csv ({pd.Series(labels).value_counts().to_dict()})", flush=True)


if __name__ == "__main__":
    main()
