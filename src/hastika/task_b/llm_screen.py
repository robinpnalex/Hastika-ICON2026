"""Which candidate LLM already reads Kanglish? An unlabelled screen, run before any
fine-tuning budget is spent.

Each model scores how well it predicts held-in training comments, as **bits per
character**. Per-token perplexity is not comparable across tokenizers (a model that
splits words finely gets low per-token loss for free), but every model has to account
for the same characters. Each comment is prefixed with a fixed English cue and only
the comment's own characters are scored, so a model whose tokenizer adds no BOS is not
penalised on its first token.

Labels are never read. Only the training portion of the holdout split is used, never
holdout or test text. Models load in 4-bit, one at a time, and each download is deleted
after scoring unless --keep-cache, because the candidates together exceed Kaggle's disk.

    python -m hastika.task_b.llm_screen --texts train.csv --out screen.json \
        --models Qwen/Qwen3-8B-Base google/gemma-4-12B
"""
import argparse
import gc
import json
import math
import os
import shutil
import time

import pandas as pd
import torch

from hastika.common.preprocessing import clean

PREFIX = "YouTube comment: "


def quant_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_use_double_quant=True)


def load_lm(name, cache_dir, token=None):
    """4-bit model with an LM head. Text-only and multimodal (Gemma-4) checkpoints
    register under different auto classes, so try them in turn."""
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(name, token=token, cache_dir=cache_dir)
    err = None
    for cls in ("AutoModelForCausalLM", "AutoModelForMultimodalLM", "AutoModelForImageTextToText"):
        auto = getattr(transformers, cls, None)
        if auto is None:
            continue
        try:
            m = auto.from_pretrained(name, quantization_config=quant_config(), dtype=torch.float16,
                                     device_map={"": 0}, token=token, cache_dir=cache_dir)
            return tok, m.eval()
        except (ValueError, KeyError, TypeError) as e:
            err = e
    raise err


@torch.no_grad()
def bits_per_char(tok, model, texts, bs=4, max_len=320):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    nll, chars = 0.0, 0
    for i in range(0, len(texts), bs):
        batch = [PREFIX + t for t in texts[i:i + bs]]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len,
                  return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        logits = model(**enc).logits[:, :-1].float()
        target = enc["input_ids"][:, 1:]
        # score a token only if it starts inside the comment, not the cue or padding
        scored = (offsets[:, 1:, 0] >= len(PREFIX)).to(model.device) & enc["attention_mask"][:, 1:].bool()
        lp = torch.log_softmax(logits, -1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
        nll -= lp[scored].sum().item()
        for j, t in enumerate(texts[i:i + bs]):
            # truncated comments: count only the characters actually covered
            covered = offsets[j][enc["attention_mask"][j].cpu().bool()][:, 1].max().item()
            chars += max(0, min(len(t), covered - len(PREFIX)))
        if not math.isfinite(nll):
            return float("nan")
    return nll / math.log(2) / max(chars, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True, help="CSV with a Comment column (training portion)")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=600, help="comments scored per model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="/tmp/hf_screen")
    ap.add_argument("--keep-cache", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or None
    df = pd.read_csv(args.texts)
    texts = [clean(t) for t in df["Comment"].sample(min(args.n, len(df)), random_state=0)]
    texts = [t for t in texts if 0 < len(t) <= 1200]
    print(f"scoring {len(texts)} comments, {sum(map(len, texts))} characters", flush=True)

    results = []
    for name in args.models:
        cache = os.path.join(args.cache, name.replace("/", "--"))
        row = {"model": name}
        t0 = time.time()
        try:
            tok, m = load_lm(name, cache, token)
            row["load_min"] = round((time.time() - t0) / 60, 1)
            torch.cuda.reset_peak_memory_stats()
            row["bpc"] = bits_per_char(tok, m, texts)
            row["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
            row["status"] = "ok" if math.isfinite(row["bpc"]) else "nan (fp16 overflow)"
            del m
        except Exception as e:      # gated, unsupported architecture, OOM: skip, keep going
            row["status"] = f"skipped: {type(e).__name__}: {str(e)[:200]}"
        gc.collect()
        torch.cuda.empty_cache()
        if not args.keep_cache:
            shutil.rmtree(cache, ignore_errors=True)
        row["minutes"] = round((time.time() - t0) / 60, 1)
        print(json.dumps(row), flush=True)
        results.append(row)
        json.dump(results, open(args.out, "w"), indent=2)

    ok = sorted((r for r in results if r["status"] == "ok"), key=lambda r: r["bpc"])
    print("\nranking, lower bits per character = reads this text better:")
    for r in ok:
        print(f"  {r['bpc']:.3f}  {r['model']}  ({r['peak_gb']} GB peak)")
    for r in results:
        if r["status"] != "ok":
            print(f"  --     {r['model']}  {r['status']}")


if __name__ == "__main__":
    main()
