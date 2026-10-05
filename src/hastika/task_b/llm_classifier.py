"""Task B with a decoder LLM as the encoder: 4-bit QLoRA plus a classification head on
the last token.

Why: on this corpus only representation-level changes have ever helped (TAPT, layer
re-initialisation), and MuRIL's wordpiece vocabulary shatters romanized Kannada. A
multilingual LLM has seen far more romanized Indic web text, and the
last-token-embedding + LoRA recipe matches or beats fine-tuned BERTs from about 1.5k
labelled rows (arXiv 2512.12677).

Design choices, each for a reason measured or known on this hardware:
  * Classification head on the last token's hidden state, not label generation:
    calibrated probabilities over the six classes, which the ensemble needs, and no
    constrained decoding.
  * The prompt names the task and the six options, so the last token's state is
    "about to answer the question" rather than "end of an arbitrary comment".
  * Fixed epoch count, no checkpoint selection on the evaluation rows; per-epoch eval
    scores are printed as diagnostics only, so the holdout stays honest.
  * Balanced class weights + label smoothing 0.05, as in the MuRIL recipe it is
    compared against, so the comparison isolates the encoder.
  * fp16 on T4 (no bf16). Non-finite losses are counted; persistent ones abort with
    exit code 3 rather than silently training garbage (Gemma is known to overflow).
  * The LM head is never loaded (AutoModel), and prepare_model_for_kbit_training is
    not used: it upcasts the embedding table to fp32, 4 GB for a 262k vocabulary.

    python -m hastika.task_b.llm_classifier --model Qwen/Qwen3-8B-Base \
        --train train.csv --eval holdout.csv --predict holdout.csv --out runs/qwen_s42
"""
import argparse
import json
import math
import os
import pathlib
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from hastika.common.preprocessing import clean

# Per task: label order (= probability column order), label column, prompt. Task A keeps
# the repo-wide [Non-Hate, Hate] column order. Prompts are short on purpose: the prompt is
# paid for on every row, every step, and Run 20's longer version roughly doubled the
# sequence length of a median comment.
TASKS = {
    "b": (["Gender", "Geo-political", "Others", "Political", "Religion", "Violence"],
          "Hate Category",
          "Kannada-English (romanized) hateful YouTube comment: {}\n"
          "Category (Gender, Political, Religion, Geo-political, Violence, Others):"),
    "a": (["Non-Hate", "Hate"], "Label",
          "Kannada-English (romanized) YouTube comment: {}\n"
          "Is this hate speech (Hate or Non-Hate)?:"),
}
LABELS, LABEL_COL, PROMPT = TASKS["b"]
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def text_backbone(m):
    """The decoder stack of a text-only or multimodal checkpoint."""
    if hasattr(m, "layers"):
        return m
    for path in ("language_model", "model.language_model", "model", "text_model"):
        obj = m
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
        except AttributeError:
            continue
        if hasattr(obj, "layers") or hasattr(obj, "embed_tokens"):
            return obj
    raise ValueError(f"no decoder stack found in {type(m).__name__}")


def load_backbone(name, token=None, cache_dir=None, compute_dtype=torch.float16):
    import transformers
    from transformers import BitsAndBytesConfig
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True)
    kw = dict(quantization_config=q, dtype=compute_dtype, device_map={"": 0}, token=token,
              cache_dir=cache_dir)
    err = None
    for cls in ("AutoModel", "AutoModelForCausalLM", "AutoModelForMultimodalLM"):
        auto = getattr(transformers, cls, None)
        if auto is None:
            continue
        try:
            return text_backbone(auto.from_pretrained(name, **kw))
        except (ValueError, KeyError, TypeError) as e:
            err = e
    raise err


class LLMClassifier(nn.Module):
    def __init__(self, backbone, n_classes, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)      # fp32, trained from scratch

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last = attention_mask.sum(1) - 1                # right padding
        pooled = h[torch.arange(h.size(0), device=h.device), last].float()
        return self.head(self.drop(pooled))


def encode(tok, texts, max_len, max_chars=700, prompt=None):
    """The comment is cut by characters before it enters the prompt, so token truncation
    (rare) can never cut off the closing "Category:" cue the head reads."""
    prompt = prompt or PROMPT
    return [tok(prompt.format(t[:max_chars]), truncation=True, max_length=max_len)["input_ids"]
            for t in texts]


def bucketed_order(lens, bs, rng, pool=50):
    """Shuffled, but batches hold similar lengths: shuffle, sort within pools of
    bs * pool rows, cut into batches, shuffle the batches. Padding is wasted compute,
    and on a T4 it was most of it."""
    order = rng.permutation(len(lens))
    out = []
    for i in range(0, len(order), bs * pool):
        chunk = sorted(order[i:i + bs * pool], key=lambda j: lens[j])
        out += [chunk[k:k + bs] for k in range(0, len(chunk), bs)]
    return np.concatenate([out[k] for k in rng.permutation(len(out))])


def batches(ids, bs, pad_id, order):
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        L = max(len(ids[j]) for j in idx)
        x = torch.full((len(idx), L), pad_id, dtype=torch.long)
        m = torch.zeros((len(idx), L), dtype=torch.long)
        for r, j in enumerate(idx):
            x[r, :len(ids[j])] = torch.tensor(ids[j])
            m[r, :len(ids[j])] = 1
        yield idx, x, m


@torch.no_grad()
def predict(model, ids, pad_id, bs, dtype):
    model.eval()
    out = np.zeros((len(ids), model.head.out_features), dtype=np.float32)
    order = np.argsort([len(x) for x in ids])          # length-sorted: less padding
    for idx, x, m in batches(ids, bs, pad_id, order):
        with torch.autocast("cuda", dtype=dtype):
            logits = model(x.cuda(), m.cuda())
        out[idx] = torch.softmax(logits.float(), -1).cpu().numpy()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", choices=["a", "b"], default="b")
    ap.add_argument("--train", required=True, help="CSV with Comment and Hate Category")
    ap.add_argument("--eval", default="", help="labelled CSV, printed per epoch, diagnostic only")
    ap.add_argument("--predict", nargs="*", default=[], help="CSVs to write probabilities for")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--lora-targets", default=",".join(LORA_TARGETS),
                    help="comma-separated module suffixes to adapt; default is attention + MLP")
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--eval-bs", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--class-weight", choices=["balanced", "sqrt", "none"], default="balanced")
    ap.add_argument("--rdrop", type=float, default=0.0,
                    help="R-Drop weight: two dropout passes tied by symmetric KL. In the 0.6410 "
                         "MuRIL recipe at 0.5; doubles the cost of a step")
    ap.add_argument("--init-adapter", default="",
                    help="start from a LoRA adapter saved by Gemma TAPT "
                         "instead of a fresh one")
    ap.add_argument("--cache", default=os.environ.get("HF_HUB_CACHE"))
    args = ap.parse_args()
    lora_targets = [x.strip() for x in args.lora_targets.split(",") if x.strip()]
    if not lora_targets:
        ap.error("--lora-targets must contain at least one module suffix")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    token = os.environ.get("HF_TOKEN") or None

    import peft.import_utils
    # peft probes torchao for every target layer and *raises* when an old one is
    # installed (Kaggle ships 0.10). It is never used here, so report it as absent.
    # This is what crashed Gemma-4-12B in Run 20 after three minutes.
    peft.import_utils.is_torchao_available = lambda: False
    try:
        import peft.tuners.lora.torchao as _lora_torchao
        _lora_torchao.is_torchao_available = lambda: False
    except ImportError:
        pass
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    labels, label_col, prompt = TASKS[args.task]
    tok = AutoTokenizer.from_pretrained(args.model, token=token, cache_dir=args.cache)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    tr = pd.read_csv(args.train)
    y = torch.tensor(tr[label_col].map(labels.index).to_numpy())
    X = encode(tok, [clean(t) for t in tr["Comment"]], args.max_len, prompt=prompt)
    lens = np.array([len(x) for x in X])
    print(f"{len(X)} training rows; prompt+comment tokens median {int(np.median(lens))}, "
          f"p99 {int(np.percentile(lens, 99))}, truncated {(lens >= args.max_len).sum()}", flush=True)
    ev = None
    if args.eval:
        ev = pd.read_csv(args.eval)
        Xev = encode(tok, [clean(t) for t in ev["Comment"]], args.max_len, prompt=prompt)
        yev = ev[label_col].map(labels.index).to_numpy()

    t0 = time.time()
    backbone = load_backbone(args.model, token, args.cache, dtype)
    backbone.config.use_cache = False
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.init_adapter:
        # TAPT applies LoRA to the same decoder stack, so the keys line up
        backbone = PeftModel.from_pretrained(backbone, args.init_adapter, is_trainable=True)
        print("initialised from TAPT adapter", args.init_adapter, flush=True)
    else:
        backbone = get_peft_model(backbone, LoraConfig(
            r=args.r, lora_alpha=2 * args.r, lora_dropout=args.lora_dropout,
            target_modules=lora_targets, bias="none"))
    print(f"LoRA rank={args.r}; target modules={lora_targets}", flush=True)
    backbone.print_trainable_parameters()
    model = LLMClassifier(backbone, len(labels))
    model.head.cuda()       # never .cuda() the whole wrapper: 4-bit weights refuse to move
    print(f"loaded in {(time.time() - t0) / 60:.1f} min; "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GB on GPU", flush=True)

    counts = np.bincount(y.numpy(), minlength=len(labels))
    w = {"balanced": len(y) / (len(labels) * counts),
         "sqrt": np.sqrt(len(y) / (len(labels) * counts)),
         "none": np.ones(len(labels))}[args.class_weight]
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w / w.mean(), dtype=torch.float32).cuda(),
                                  label_smoothing=args.label_smoothing)

    lora = [p for n, p in model.named_parameters() if p.requires_grad and "head" not in n.split(".")[0]]
    opt = torch.optim.AdamW([{"params": lora, "lr": args.lr},
                             {"params": model.head.parameters(), "lr": args.head_lr}],
                            weight_decay=args.weight_decay)
    steps = math.ceil(len(X) / args.bs / args.grad_accum) * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(args.warmup * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    torch.cuda.reset_peak_memory_stats()

    bad, step, history = 0, 0, []
    rng = np.random.default_rng(args.seed)
    t_train = time.time()
    model.train()
    for epoch in range(args.epochs):
        order = bucketed_order(lens, args.bs, rng)
        run_loss, n = 0.0, 0
        opt.zero_grad(set_to_none=True)     # drop any partial accumulation window
        for b, (idx, x, m) in enumerate(batches(X, args.bs, pad_id, order)):
            with torch.autocast("cuda", dtype=dtype):
                logits = model(x.cuda(), m.cuda())
            target = y[idx].cuda()
            loss = loss_fn(logits.float(), target)
            if args.rdrop:
                # second pass sees different dropout masks (LoRA and head); the symmetric
                # KL pulls the two predictive distributions together
                with torch.autocast("cuda", dtype=dtype):
                    logits2 = model(x.cuda(), m.cuda())
                lp, lq = torch.log_softmax(logits.float(), -1), torch.log_softmax(logits2.float(), -1)
                kl = 0.5 * (nn.functional.kl_div(lp, lq, log_target=True, reduction="batchmean")
                            + nn.functional.kl_div(lq, lp, log_target=True, reduction="batchmean"))
                loss = 0.5 * (loss + loss_fn(logits2.float(), target)) + args.rdrop * kl
            loss = loss / args.grad_accum
            if not torch.isfinite(loss):
                bad += 1
                if bad > 20:
                    print("ABORT: more than 20 non-finite losses -- fp16 overflow in this model",
                          flush=True)
                    sys.exit(3)
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            run_loss += loss.item() * args.grad_accum
            n += 1
            if (b + 1) % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % 25 == 0:
                    el = time.time() - t_train
                    print(f"epoch {epoch + 1} step {step}/{steps} loss {run_loss / max(n, 1):.4f} "
                          f"{el / 60:.1f} min, eta {el / step * (steps - step) / 60:.0f} min, "
                          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        rec = {"epoch": epoch + 1, "train_loss": run_loss / max(n, 1), "non_finite": bad}
        if ev is not None:
            p = predict(model, Xev, pad_id, args.eval_bs, dtype)
            rec["eval_macro_f1"] = float(f1_score(yev, p.argmax(1), average="macro"))
            print(f"== epoch {epoch + 1}: eval macro-F1 {rec['eval_macro_f1']:.4f} "
                  "(diagnostic only, not used for selection)", flush=True)
        history.append(rec)

    for path in args.predict:
        df = pd.read_csv(path)
        p = predict(model, encode(tok, [clean(t) for t in df["Comment"]], args.max_len,
                                  prompt=prompt), pad_id, args.eval_bs, dtype)
        stem = pathlib.Path(path).stem
        np.save(out / f"{stem}_probs.npy", p)
        df[["id"]].to_csv(out / f"{stem}_ids.csv", index=False)
        print(f"wrote {out / (stem + '_probs.npy')} {p.shape}", flush=True)
    json.dump({"args": vars(args), "history": history, "minutes": (time.time() - t0) / 60,
               "peak_gb": torch.cuda.max_memory_allocated() / 2**30},
              open(out / "metrics.json", "w"), indent=2)
    print(f"finished in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
