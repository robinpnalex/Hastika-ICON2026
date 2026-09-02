#!/usr/bin/env python3
"""Generate a HASTIKA Task A predictions.csv from a fine-tuned checkpoint."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from finetune_task_a import ID2LABEL, LABEL2ID, clean_text, demojize_text


class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, index):
        return {
            key: torch.tensor(value[index], dtype=torch.long)
            for key, value in self.encodings.items()
        }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default="checkpoints/muril_task_a",
        help="Directory containing the saved model and tokenizer.",
    )
    parser.add_argument(
        "--input-csv",
        default="data/binary_validation_inputs.csv",
        help="Task A input CSV containing id and Comment columns.",
    )
    parser.add_argument(
        "--output",
        default="predictions.csv",
        help="Submission CSV to create.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Override the training maximum length.",
    )
    parser.add_argument(
        "--text-mode",
        choices=["auto", "clean", "raw"],
        default="auto",
        help="Auto reuses the cleaning setting stored during training.",
    )
    parser.add_argument(
        "--emoji-mode",
        choices=["auto", "preserve", "demojize"],
        default="auto",
        help="Auto reuses the emoji setting stored during training.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use CUDA mixed precision during inference.",
    )
    return parser.parse_args()


def read_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"No rows found in {path}")

    columns = list(rows[0])
    id_key = next(
        (column for column in columns if column.strip().lower() in {"id", "index"}),
        None,
    )
    comment_key = next(
        (column for column in columns if column.strip().lower() == "comment"),
        None,
    )
    if id_key is None or comment_key is None:
        raise ValueError(f"Expected id and Comment columns in {path}; found {columns}")
    return rows, id_key, comment_key


def read_training_metadata(model_dir: Path):
    metadata_path = model_dir / "training_metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_label(model, prediction_id: int) -> str:
    configured = model.config.id2label.get(
        prediction_id,
        model.config.id2label.get(str(prediction_id)),
    )
    if configured in LABEL2ID:
        return configured
    return ID2LABEL[prediction_id]


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(f"Checkpoint directory does not exist: {model_dir}")

    if args.fp16 and not torch.cuda.is_available():
        raise SystemExit("--fp16 requires a CUDA GPU.")

    metadata = read_training_metadata(model_dir)
    max_length = args.max_length or int(metadata.get("max_length", 160))
    if args.text_mode == "auto":
        use_clean_text = bool(metadata.get("cleaned_text", True))
    else:
        use_clean_text = args.text_mode == "clean"
    if args.emoji_mode == "auto":
        use_demojized_text = bool(metadata.get("demojized", False))
    else:
        use_demojized_text = args.emoji_mode == "demojize"

    rows, id_key, comment_key = read_rows(Path(args.input_csv))
    texts = [
        clean_text(row[comment_key]) if use_clean_text else row[comment_key]
        for row in rows
    ]
    if use_demojized_text:
        texts = [demojize_text(text) for text in texts]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Rows: {len(rows)}, max_length: {max_length}, "
        f"text mode: {'clean' if use_clean_text else 'raw'}, "
        f"emoji mode: {'demojize' if use_demojized_text else 'preserve'}"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    dataset = InferenceDataset(texts, tokenizer, max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
        pin_memory=device.type == "cuda",
    )

    prediction_ids = []
    use_amp = args.fp16 and device.type == "cuda"
    with torch.inference_mode():
        for batch in loader:
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(**batch).logits
            prediction_ids.extend(logits.argmax(dim=-1).cpu().tolist())

    predictions = [
        resolve_label(model, prediction_id) for prediction_id in prediction_ids
    ]
    if len(predictions) != len(rows):
        raise RuntimeError(
            f"Prediction count mismatch: expected {len(rows)}, got {len(predictions)}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label"])
        writer.writerows(
            (row[id_key], prediction) for row, prediction in zip(rows, predictions)
        )

    print(f"Prediction distribution: {dict(Counter(predictions))}")
    print(f"Wrote {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
