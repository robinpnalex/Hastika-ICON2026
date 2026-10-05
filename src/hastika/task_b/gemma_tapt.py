"""Causal-language-model TAPT for Gemma before the Task B classifier.

This is deliberately separate from ``llm_classifier``: the latter uses AutoModel and
does not load Gemma's large language-model head, while TAPT needs the causal-LM loss.
The saved PEFT adapter can be passed to ``llm_classifier --init-adapter``.

Example::

    python -m hastika.task_b.gemma_tapt --model google/gemma-4-12B \
        --train data/derived/task_b_combined_holdout/train.csv \
        --out artifacts/runs/gemma_tapt_s42 --seed 42
"""
import argparse
import json
import math
import os
import pathlib
import random
import time

import numpy as np
import pandas as pd
import torch

from hastika.common.preprocessing import clean
from hastika.task_b.llm_classifier import LORA_TARGETS, PROMPT


def batches(ids, bs, pad_id, order):
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        length = max(len(ids[j]) for j in idx)
        x = torch.full((len(idx), length), pad_id, dtype=torch.long)
        mask = torch.zeros((len(idx), length), dtype=torch.long)
        for row, j in enumerate(idx):
            x[row, :len(ids[j])] = torch.tensor(ids[j])
            mask[row, :len(ids[j])] = 1
        yield idx, x, mask


def bucketed_order(lengths, bs, rng, pool=50):
    order = rng.permutation(len(lengths))
    chunks = []
    for start in range(0, len(order), bs * pool):
        chunk = sorted(order[start:start + bs * pool], key=lambda j: lengths[j])
        chunks += [chunk[k:k + bs] for k in range(0, len(chunk), bs)]
    return np.concatenate([chunks[k] for k in rng.permutation(len(chunks))])


def load_lm(name, token, cache, dtype):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, quantization_config=quant, dtype=dtype, device_map={"": 0},
        token=token, cache_dir=cache)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--cache", default=os.environ.get("HF_HUB_CACHE"))
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = (torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8
             else torch.float16)
    token = os.environ.get("HF_TOKEN") or None

    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
    try:
        import peft.tuners.lora.torchao as _lora_torchao
        _lora_torchao.is_torchao_available = lambda: False
    except ImportError:
        pass
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    tok = AutoTokenizer.from_pretrained(args.model, token=token, cache_dir=args.cache)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    df = pd.read_csv(args.train)
    texts = [PROMPT.format(clean(t)) for t in df["Comment"]]
    ids = [tok(t, truncation=True, max_length=args.max_len)["input_ids"] for t in texts]
    lengths = np.array([len(x) for x in ids])
    print(f"{len(ids)} TAPT rows; median tokens {int(np.median(lengths))}; "
          f"p99 {int(np.percentile(lengths, 99))}; truncated {(lengths >= args.max_len).sum()}",
          flush=True)

    t0 = time.time()
    model = load_lm(args.model, token, args.cache, dtype)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # Leave task_type unset intentionally. The adapter is trained through the
    # causal-LM wrapper here, then loaded into AutoModel by llm_classifier, whose
    # pooled representation comes from last_hidden_state rather than logits.
    model = get_peft_model(model, LoraConfig(
        r=args.r, lora_alpha=2 * args.r, lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGETS, bias="none"))
    model.print_trainable_parameters()
    print(f"loaded in {(time.time() - t0) / 60:.1f} min", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    steps = math.ceil(len(ids) / args.bs / args.grad_accum) * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(args.warmup * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    rng = np.random.default_rng(args.seed)
    history, bad, step = [], 0, 0
    t_train = time.time()
    model.train()
    for epoch in range(args.epochs):
        order = bucketed_order(lengths, args.bs, rng)
        total, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for batch_no, (_idx, x, mask) in enumerate(batches(ids, args.bs, pad_id, order)):
            labels = x.masked_fill(mask == 0, -100)
            with torch.autocast("cuda", dtype=dtype):
                loss = model(input_ids=x.cuda(), attention_mask=mask.cuda(),
                             labels=labels.cuda()).loss
            loss = loss / args.grad_accum
            if not torch.isfinite(loss):
                bad += 1
                opt.zero_grad(set_to_none=True)
                if bad > 20:
                    raise RuntimeError("more than 20 non-finite TAPT losses")
                continue
            scaler.scale(loss).backward()
            total += loss.item() * args.grad_accum
            n += 1
            if (batch_no + 1) % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % 25 == 0:
                    elapsed = time.time() - t_train
                    print(f"epoch {epoch + 1} step {step}/{steps} loss {total / max(n, 1):.4f} "
                          f"{elapsed / 60:.1f} min", flush=True)
        rec = {"epoch": epoch + 1, "loss": total / max(n, 1), "non_finite": bad}
        history.append(rec)
        print(f"TAPT epoch {epoch + 1}: loss {rec['loss']:.4f}", flush=True)

    adapter = out / "adapter"
    model.save_pretrained(adapter)
    tok.save_pretrained(adapter)
    json.dump({"args": vars(args), "history": history,
               "minutes": (time.time() - t0) / 60,
               "adapter": str(adapter)}, open(out / "metrics.json", "w"), indent=2)
    print(f"saved TAPT adapter to {adapter}", flush=True)


if __name__ == "__main__":
    main()
