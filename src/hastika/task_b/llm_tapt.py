"""Task-adaptive pretraining (TAPT) for a decoder LLM: LoRA next-token training on
unlabelled Kanglish comments, saved as an adapter the classifier starts from.

The MuRIL lesson: TAPT was worth +2.9 on Task B and +2.3 on Task A, the largest single
gains before the LLM. For MuRIL it was masked-LM on the comments. For a decoder it is
causal LM: predict each token of a comment from the ones before it. That teaches the
romanized spellings and code-mixing of this corpus before any label is seen.

LoRA goes on the *same* decoder stack, with the same target modules and rank, that
llm_classifier.py adapts. The saved adapter's keys therefore line up exactly, and
`llm_classifier.py --init-adapter <out>` continues from it.

Only unlabelled text is read. Pass training-portion comments and external Kannada text;
never holdout or test comments.

    python -m hastika.task_b.llm_tapt --model google/gemma-4-12B \
        --texts train.csv data/external/offenseval_kn.csv --out runs/tapt_adapter
"""
import argparse
import math
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from hastika.common.preprocessing import clean
from hastika.task_b.llm_classifier import LORA_TARGETS, batches, bucketed_order, text_backbone
from hastika.task_b.llm_screen import load_lm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--texts", nargs="+", required=True, help="CSVs with a Comment column")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16)
    # micro-batch 4: the LM loss materialises logits over a 262k vocabulary, about 1.3 GB
    # of fp32 per 4 x 160 tokens plus its gradient, on top of the 4-bit 12B model
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--logit-chunk", type=int, default=64,
                    help="hidden-state positions projected through the LM head at once")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", default=os.environ.get("HF_HUB_CACHE"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    token = os.environ.get("HF_TOKEN") or None

    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False     # see llm_classifier.py
    try:
        import peft.tuners.lora.torchao as _lora_torchao
        _lora_torchao.is_torchao_available = lambda: False
    except ImportError:
        pass
    from peft import LoraConfig, get_peft_model
    from transformers import get_cosine_schedule_with_warmup

    texts = []
    for path in args.texts:
        texts += [clean(t) for t in pd.read_csv(path)["Comment"].astype(str)]
    texts = list(dict.fromkeys(t for t in texts if len(t.split()) >= 2))   # dedupe, keep order
    print(f"{len(texts)} unique comments of 2+ words from {len(args.texts)} files", flush=True)

    t0 = time.time()
    tok, model = load_lm(args.model, args.cache, token)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    lm_head = model.get_output_embeddings()
    cfg = getattr(model.config, "text_config", model.config)
    softcap = getattr(cfg, "final_logit_softcapping", None)
    backbone = text_backbone(model)
    backbone.config.use_cache = False
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    backbone = get_peft_model(backbone, LoraConfig(
        r=args.r, lora_alpha=2 * args.r, lora_dropout=0.05, target_modules=LORA_TARGETS,
        bias="none"))
    backbone.print_trainable_parameters()
    for p in lm_head.parameters():
        p.requires_grad_(False)
    print(f"loaded in {(time.time() - t0) / 60:.1f} min, softcap {softcap}", flush=True)

    ids = [tok(t, truncation=True, max_length=args.max_len)["input_ids"] for t in texts]
    lens = np.array([len(x) for x in ids])
    params = [p for p in backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps = math.ceil(len(ids) / args.bs / args.grad_accum) * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    rng = np.random.default_rng(args.seed)
    step, bad, t_train = 0, 0, time.time()
    for epoch in range(args.epochs):
        run, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for b, (_, x, m) in enumerate(batches(ids, args.bs, pad_id, bucketed_order(lens, args.bs, rng))):
            x, m = x.cuda(), m.cuda()
            target = x[:, 1:].masked_fill(m[:, 1:] == 0, -100)
            with torch.autocast("cuda", dtype=dtype):
                h = backbone(input_ids=x, attention_mask=m).last_hidden_state[:, :-1]
            flat_h = h.reshape(-1, h.size(-1))
            flat_target = target.reshape(-1)
            valid = flat_target.ne(-100)
            n_valid = int(valid.sum())
            if n_valid == 0:
                continue
            batch_loss = 0.0
            # The LM head has a ~262k vocabulary. Projecting all B*T positions at
            # once creates a several-hundred-MB logits tensor on top of the 12B
            # backbone. Backpropagate each small position chunk through the shared
            # hidden-state graph instead; this preserves the exact token loss while
            # keeping the logits and softmax workspace bounded.
            chunks = list(range(0, flat_h.size(0), args.logit_chunk))
            for chunk_no, start in enumerate(chunks):
                end = min(start + args.logit_chunk, flat_h.size(0))
                if not valid[start:end].any():
                    continue
                with torch.autocast("cuda", dtype=dtype):
                    logits = lm_head(flat_h[start:end]).float()
                    if softcap:
                        logits = torch.tanh(logits / softcap) * softcap
                chunk_loss = nn.functional.cross_entropy(
                    logits, flat_target[start:end], ignore_index=-100, reduction="sum") / n_valid
                if not torch.isfinite(chunk_loss):
                    bad += 1
                    if bad > 20:
                        print("ABORT: more than 20 non-finite losses -- fp16 overflow", flush=True)
                        sys.exit(3)
                    opt.zero_grad(set_to_none=True)
                    break
                retain = chunk_no < len(chunks) - 1
                scaler.scale(chunk_loss / args.grad_accum).backward(retain_graph=retain)
                batch_loss += chunk_loss.item()
                del logits, chunk_loss
            else:
                run, n = run + batch_loss, n + 1
            if (b + 1) % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % 25 == 0:
                    el = time.time() - t_train
                    print(f"epoch {epoch + 1} step {step}/{steps} lm loss {run / n:.4f} "
                          f"{el / 60:.1f} min, eta {el / step * (steps - step) / 60:.0f} min, "
                          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        print(f"== epoch {epoch + 1}: mean lm loss {run / max(n, 1):.4f}", flush=True)

    out = pathlib.Path(args.out)
    backbone.save_pretrained(out)
    print(f"saved adapter to {out} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
