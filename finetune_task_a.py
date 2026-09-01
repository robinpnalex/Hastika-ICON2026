#!/usr/bin/env python3
"""
Fine-tune MuRIL or XLM-R for HASTIKA Task A.

Example:
    python finetune_task_a.py \
        --model google/muril-base-cased \
        --output-dir checkpoints/muril_task_a

The released validation inputs do not contain labels, so this script creates a
stratified validation split from binary_train.csv for model selection.
"""

import argparse
import csv
import html
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    set_seed,
)


LABEL2ID = {"Non-Hate": 0, "Hate": 1}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}


def repair_mojibake(text: str) -> str:
    """Repair common UTF-8-as-Latin-1 corruption without touching valid Unicode."""
    for _ in range(2):
        try:
            repaired = text.encode("latin1").decode("utf8")
        except UnicodeError:
            break

        markers = ("à", "ð", "â", "Ã", "Â")
        before = sum(text.count(marker) for marker in markers)
        after = sum(repaired.count(marker) for marker in markers)
        if after < before:
            text = repaired
        else:
            break
    return text


def clean_text(text: str) -> str:
    text = repair_mojibake(text)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: torch.tensor(value[index], dtype=torch.long)
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def make_collate_fn(tokenizer):
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def collate(features):
        labels = torch.tensor(
            [feature.pop("labels").item() for feature in features],
            dtype=torch.long,
        )
        batch = collator(features)
        batch["labels"] = labels
        return batch

    return collate


def read_training_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    required = {"Comment", "Label"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    invalid = sorted({row["Label"] for row in rows} - set(LABEL2ID))
    if invalid:
        raise ValueError(f"Unexpected labels: {invalid}")
    return rows


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def evaluate(model, loader, device):
    model.eval()
    predictions = []
    labels = []

    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(
                **{
                    key: value
                    for key, value in batch.items()
                    if key != "labels"
                }
            )
            predictions.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())

    return {
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "accuracy": accuracy_score(labels, predictions),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="google/muril-base-cased",
        help="Hugging Face encoder checkpoint.",
    )
    parser.add_argument(
        "--train-csv",
        default="data/binary_train.csv",
        help="Task A labeled training CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/task_a",
        help="Directory for the best model and tokenizer.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help="Task A is nearly balanced; start with none.",
    )
    parser.add_argument(
        "--raw-text",
        action="store_true",
        help="Use original comments instead of mojibake/HTML cleaning.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use CUDA mixed precision. Recommended on a supported NVIDIA GPU.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.fp16 and not torch.cuda.is_available():
        raise SystemExit("--fp16 requires a CUDA GPU.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    rows = read_training_rows(Path(args.train_csv))
    texts = [
        row["Comment"] if args.raw_text else clean_text(row["Comment"])
        for row in rows
    ]
    labels = [LABEL2ID[row["Label"]] for row in rows]
    indices = np.arange(len(rows))

    train_indices, valid_indices = train_test_split(
        indices,
        test_size=args.valid_size,
        stratify=labels,
        random_state=args.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    train_dataset = CommentDataset(
        [texts[index] for index in train_indices],
        [labels[index] for index in train_indices],
        tokenizer,
        args.max_length,
    )
    valid_dataset = CommentDataset(
        [texts[index] for index in valid_indices],
        [labels[index] for index in valid_indices],
        tokenizer,
        args.max_length,
    )
    collate_fn = make_collate_fn(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    if args.class_weight == "balanced":
        counts = np.bincount(
            [labels[index] for index in train_indices], minlength=2
        )
        weights = len(train_indices) / (2.0 * counts)
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device)
        )
        print(f"Using class weights: {weights.tolist()}")
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_f1 = -1.0
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = move_batch(batch, device)
            labels_batch = batch.pop("labels")

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(**batch).logits
                loss = loss_fn(logits, labels_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        metrics = evaluate(model, valid_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            **metrics,
        }
        history.append(record)
        print(
            f"epoch={epoch} loss={record['train_loss']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"accuracy={metrics['accuracy']:.4f}"
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            # Keep the checkpoint on CPU to avoid doubling GPU model memory.
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break

    if best_state is None:
        raise RuntimeError("No checkpoint was produced.")

    model.load_state_dict(best_state)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "model": args.model,
        "seed": args.seed,
        "max_length": args.max_length,
        "cleaned_text": not args.raw_text,
        "best_macro_f1": best_f1,
        "history": history,
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved best model to {output_dir}")


if __name__ == "__main__":
    main()
