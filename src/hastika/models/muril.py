"""Best-practice MuRIL fine-tune for HASTIKA Task A (binary hate speech).

Every non-default choice here is justified by a measurement on this dataset --
see docs/task_a/RESEARCH.md. Summary of what drove the design:

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
  * --external adds the OffensEval-Dravidian Kannada corpus.
    Its labels are noticeably noisier than HASTIKA's, so the default is to consume
    it as a separate first stage rather than to mix it into the folds -- see
    load_external() for why.

Outputs are written in the same layout train_xlmr.py uses, so ensemble.py can
blend these runs with the SVM and the other encoders.
"""
import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from hastika.common.paths import PROJECT_ROOT, RAW_DATA_DIR, RUNS_DIR
from hastika.common.preprocessing import clean, dedupe_index
from hastika.task_b import axes

ROOT = PROJECT_ROOT
RUNS = RUNS_DIR
SPLIT_SEED = 42          # must match baseline_svm.py so OOF rows line up
N_SPLITS = 5


# ---------------------------------------------------------------- data

class Comments(Dataset):
    def __init__(self, texts, labels=None, aux=None):
        self.texts = list(texts)
        self.labels = None if labels is None else list(labels)
        self.aux = None if aux is None else list(aux)      # auxiliary act-axis target

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return (self.texts[i],
                -1 if self.labels is None else self.labels[i],
                -1 if self.aux is None else self.aux[i])


def make_collate(tok, max_len, remap=None):
    """remap: LongTensor[orig_vocab] -> trimmed ids, or None to keep the full vocab."""
    def collate(batch):
        texts, labels, aux = zip(*batch)
        enc = tok(list(texts), truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt")
        if remap is not None:
            enc["input_ids"] = remap[enc["input_ids"]]
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        if aux[0] != -1:
            enc["aux_labels"] = torch.tensor(aux, dtype=torch.long)
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

    def __init__(self, name, pooling="meanmax", dropout=0.1, reinit_layers=0, n_classes=2,
                 keep_ids=None, n_aux=0):
        super().__init__()
        cfg = AutoConfig.from_pretrained(name)
        cfg.hidden_dropout_prob = dropout
        cfg.attention_probs_dropout_prob = dropout
        self.backbone = AutoModel.from_pretrained(name, config=cfg)
        if keep_ids is not None:
            self._trim_vocab(keep_ids)
        self.pooling = pooling
        h = cfg.hidden_size
        width = {"cls": h, "mean": h, "meanmax": 2 * h, "last4": 4 * h}[pooling]
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width, n_classes)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        # Auxiliary head over the SAME pooled vector. It never touches the main
        # logits; its only job is to push the encoder to represent the act axis.
        self.aux_head = nn.Linear(width, n_aux) if n_aux else None
        if self.aux_head is not None:
            nn.init.normal_(self.aux_head.weight, std=0.02)
            nn.init.zeros_(self.aux_head.bias)
        if reinit_layers:
            self._reinit_top(reinit_layers)

    def _trim_vocab(self, keep_ids):
        """Drop embedding rows for tokens this corpus never produces.

        MuRIL ships a 197k wordpiece vocab for 17 languages; the HASTIKA CSVs
        touch 9,264 of them (4.7%). The other 95% are dead rows that AdamW still
        carries two moments for, and that FGM clones in full on every step --
        together the single largest term in peak RSS. Holding out an entire
        released file, 0.178% of its tokens fall outside a vocab built from the
        rest, so wordpiece backoff covers the unseen test set.
        """
        emb = self.backbone.embeddings.word_embeddings
        assert keep_ids[0] == 0, "pad id 0 must be kept first so padding_idx stays 0"
        trimmed = nn.Embedding(len(keep_ids), emb.embedding_dim, padding_idx=0)
        trimmed.weight.data = emb.weight.data[keep_ids].clone()
        self.backbone.embeddings.word_embeddings = trimmed
        self.backbone.config.vocab_size = len(keep_ids)

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

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                return_aux=False, **_):
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

        logits = self.head(self.dropout(pooled))
        if return_aux and self.aux_head is not None:
            return logits, self.aux_head(self.dropout(pooled))
        return logits


# ------------------------------------------------- training utilities

def make_loss(args):
    """Cross-entropy, or focal loss when --loss focal.

    Focal down-weights examples the model already gets right, by (1-p)^gamma.
    On Task B that targets the confusable middle rather than whole classes:
    17% of rows carry two or more competing topic cues, and class weighting
    cannot see that -- it reweights Geo-political uniformly whether the row is
    obvious or contested. The two compose, so --class-weight balanced stays on.
    """
    weight = getattr(args, "class_weight_t", None)
    smooth = args.label_smoothing
    gamma = getattr(args, "focal_gamma", 0.0)
    if getattr(args, "loss", "ce") != "focal" or gamma <= 0:
        return lambda logits, y: F.cross_entropy(logits, y, label_smoothing=smooth,
                                                 weight=weight)

    def focal(logits, y):
        # per-example CE keeps label smoothing and class weights intact, then the
        # focal factor scales each term by how confident the model already is
        ce = F.cross_entropy(logits, y, label_smoothing=smooth, weight=weight,
                             reduction="none")
        pt = F.softmax(logits.detach(), -1).gather(1, y[:, None]).squeeze(1)
        loss = ((1 - pt) ** gamma) * ce
        if weight is None:
            return loss.mean()
        # weighted CE normalises by summed weights, so match that
        return loss.sum() / weight[y].sum().clamp(min=1e-9)

    return focal


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
    """Exponential moving average of weights; evaluated instead of the raw model.

    Bias-corrected, Adam-style. Without the correction the shadow keeps a
    d**t share of the *initialization* forever, and on a run this short that
    share is not small: Task B trains 942 optimizer steps per fold, so at
    decay 0.999 the evaluated weights are 39% random init (0.999**942 = 0.39).
    Task A hides the problem because it takes ~2400 steps (0.09). Dividing by
    1 - d**t makes the shadow a proper weighted average of the trajectory so
    far, which is what "moving average of weights" was supposed to mean, and
    leaves the long-run behaviour unchanged.
    """

    def __init__(self, model, decay=0.999, bias_correct=True):
        self.decay = decay
        self.bias_correct = bias_correct
        self.t = 0
        self.shadow = {k: torch.zeros_like(v) if bias_correct else v.detach().clone()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        self.t += 1
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def state_dict_for_eval(self, model):
        """Model's own state dict with every float tensor replaced by the shadow.

        Deliberately not a deepcopy: load_state_dict copies into the model's
        parameters, so the caller's separate raw snapshot is what preserves the
        live weights. Copying the full 293M-parameter dict here as well was one
        wasted gigabyte per evaluation, twelve times a fold.
        """
        sd = dict(model.state_dict())
        if not self.bias_correct or self.t == 0:
            sd.update(self.shadow)
            return sd
        scale = 1.0 / (1.0 - self.decay ** self.t)
        sd.update({k: v * scale for k, v in self.shadow.items()})
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


# ---------------------------------------------------------------- external data

EXTERNAL_DEFAULT = ROOT / "data" / "external" / "offenseval_kn.csv"


def load_external(path, demoji, drop_other=True):
    """Auxiliary Kannada-English hate corpus, built by task_b.fetch_external.

    That script documents the source, the six-class -> binary mapping, and the
    filtering (native-script rows and untargeted profanity dropped, ids prefixed
    `ext_`, text deduped against every HASTIKA comment we hold).

    Two properties drive how it is used here. Its labels are noisier than
    HASTIKA's -- spot-checking Hate rows turns up plain mislabels, worst in the
    `_Other` subclass, hence drop_other -- and it is 4x more Non-Hate than Hate,
    against HASTIKA's near-even split. Both argue for --external-mode stage:
    a second stage on clean, balanced labels overwrites the boundary this corpus
    sets, whereas mixing leaves its noise and its skew in the final loss.
    """
    df = pd.read_csv(path)
    if drop_other and "source_label" in df.columns:
        df = df[df["source_label"] != "Offensive_Targeted_Insult_Other"]
    X = df["Comment"].map(lambda t: clean(t, demojize=demoji)).values
    y = (df["Label"] == "Hate").astype(int).values
    return X, y


# ---------------------------------------------------------------- train

def train_fold(args, tok, X_tr, y_tr, X_va, y_va, X_test, device, tag, return_state=False):
    torch.manual_seed(args.seed)
    aux_w = float(getattr(args, "aux_weight", 0.0))
    model = MurilClassifier(args.model, args.pooling, args.dropout, args.reinit_layers,
                            n_classes=getattr(args, "n_classes", 2),
                            keep_ids=getattr(args, "keep_ids", None),
                            n_aux=2 if aux_w > 0 else 0).to(device)

    # Warm start from a previous stage. This overwrites the freshly re-initialized
    # top layers, which is the intent: that stage already trained them. The head is
    # re-initialized by default because it was fitted to the auxiliary corpus's
    # label convention, not this one.
    init = getattr(args, "init_state", None)
    if init is not None:
        model.load_state_dict({k: v.to(device) for k, v in init.items()})
        if getattr(args, "external_reinit_head", True):
            nn.init.normal_(model.head.weight, std=0.02)
            nn.init.zeros_(model.head.bias)
    # --folds 1 trains on every labelled row. There is then no held-out set to
    # select a checkpoint against, so the final weights are taken and no score is
    # reported. Nothing else in the file passes an empty validation set.
    no_val = len(y_va) == 0
    collate = make_collate(tok, args.max_len, getattr(args, "remap", None))
    aux_tr = axes.aux_labels(X_tr) if aux_w > 0 else None
    if aux_tr is not None:
        print(f"  [{tag}] aux act-axis targets: {sum(aux_tr)}/{len(aux_tr)} positive, "
              f"weight {aux_w}", flush=True)
    tr = DataLoader(Comments(X_tr, y_tr, aux_tr), batch_size=args.bs, shuffle=True,
                    collate_fn=collate, drop_last=True)
    va = DataLoader(Comments(X_va, y_va), batch_size=args.eval_bs, collate_fn=collate)
    te = DataLoader(Comments(X_test), batch_size=args.eval_bs, collate_fn=collate)

    opt = torch.optim.AdamW(
        llrd_param_groups(model, args.lr, args.head_lr, args.llrd, args.weight_decay), eps=1e-6)
    steps = max(1, (len(tr) // args.grad_accum) * args.epochs)
    sched = get_cosine_schedule_with_warmup(opt, int(args.warmup * steps), steps)
    dtype = amp_dtype(device, args.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    criterion = make_loss(args)
    fgm = FGM(model, args.fgm_eps) if args.fgm else None
    ema = (EMA(model, args.ema_decay, getattr(args, 'ema_bias_correct', True))
           if args.ema else None)

    # evaluate this many times per epoch -- epoch-level granularity is too coarse
    # given how sharply val F1 moves between epochs on this dataset
    eval_every = max(1, len(tr) // args.evals_per_epoch)
    best = {"f1": -1.0, "va": None, "te": None}
    # The last evaluation is kept as well as the best one. Picking the best
    # checkpoint by macro-F1 on the validation fold and then reporting that same
    # fold as out-of-fold is selection on the test rows: the number it produces is
    # optimistic by however much twelve draws of eval noise are worth. Carrying
    # the final checkpoint costs one extra test-set prediction per fold and makes
    # the size of that optimism visible in every run. --select last uses it.
    last = {"f1": -1.0, "va": None, "te": None}
    step_i = 0

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        for step, batch in enumerate(tr, 1):
            model.train()
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            aux_y = batch.pop("aux_labels", None)

            with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                if aux_y is not None and model.aux_head is not None:
                    logits, aux_logits = model(**batch, return_aux=True)
                else:
                    logits, aux_logits = model(**batch), None
                loss = criterion(logits, labels)
                if args.rdrop:
                    logits2 = model(**batch)
                    loss = 0.5 * (loss + criterion(logits2, labels))
                    kl = 0.5 * (F.kl_div(F.log_softmax(logits, -1), F.softmax(logits2, -1),
                                         reduction="batchmean")
                                + F.kl_div(F.log_softmax(logits2, -1), F.softmax(logits, -1),
                                           reduction="batchmean"))
                    loss = loss + args.rdrop * kl
                if aux_logits is not None:
                    # weak supervision on the act axis: 443 positives against the
                    # Violence class's 221, so the encoder gets twice the signal
                    # for "a threat is present" without touching the main head
                    loss = loss + aux_w * F.cross_entropy(aux_logits.float(), aux_y)
            scaler.scale(loss / args.grad_accum).backward()

            if fgm is not None and fgm.attack():
                with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                    adv = criterion(model(**batch), labels)
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
                final = (ep == args.epochs and step == len(tr))
                if no_val and not final:
                    continue          # nothing to score, and only the end matters
                if ema is not None:
                    raw = copy.deepcopy(model.state_dict())
                    model.load_state_dict(ema.state_dict_for_eval(model))
                if no_val:
                    p_va, f1 = None, float("nan")
                else:
                    p_va = predict(model, va, device, dtype)
                    f1 = f1_score(y_va, p_va.argmax(1), average="macro")
                improved = f1 > best["f1"]
                if improved or final:
                    p_te = predict(model, te, device, dtype)
                    if improved:
                        best = {"f1": f1, "va": p_va, "te": p_te}
                        if return_state:
                            # snapshot on CPU: an 8GB card has no room for a second copy
                            best["state"] = {k: v.detach().cpu().clone()
                                             for k, v in model.state_dict().items()}
                    if final:
                        last = {"f1": f1, "va": p_va, "te": p_te}
                        if return_state:
                            last["state"] = {k: v.detach().cpu().clone()
                                             for k, v in model.state_dict().items()}
                mark = " *" if improved else ""
                if ema is not None:
                    model.load_state_dict(raw)
                shown = "  no val" if no_val else f"val {f1:.4f}{mark}"
                print(f"  [{tag}] ep{ep} {step}/{len(tr)} loss {loss.item():.4f} "
                      f"{shown} ({(time.time()-t0)/step:.2f}s/step)", flush=True)

        tail = "no val" if no_val else f"best {best['f1']:.4f}"
        print(f"  [{tag}] ep{ep} done {(time.time()-t0)/60:.1f}min  {tail}", flush=True)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # with no validation set `best` was never filled, so `last` is the only option
    want_last = no_val or (getattr(args, "select", "best") == "last"
                           and last["va"] is not None)
    pick, alt = (last, best) if want_last else (best, last)
    # The checkpoint that was not selected, kept so a single run can report both
    # the optimistic and the unbiased OOF instead of needing two.
    args.alt_fold = alt
    if no_val:
        print(f"  [{tag}] trained on every row; final checkpoint taken, no score",
              flush=True)
    else:
        print(f"  [{tag}] best {best['f1']:.4f}  last {last['f1']:.4f}  "
              f"(selected: {getattr(args, 'select', 'best')})", flush=True)
    if return_state:
        return pick["f1"], pick["va"], pick["te"], pick.get("state")
    return pick["f1"], pick["va"], pick["te"]


def with_external(X_tr, y_tr, X_ext, y_ext, args):
    """Append the external rows to a TRAINING fold under --external-mode mix.

    Only ever called on the training side. External rows entering a validation
    fold would measure fit to the auxiliary corpus, which is not a quantity
    anyone wants -- the same discipline experiments/task_a/synthetic/evaluate.py applies.
    """
    if X_ext is None or args.external_mode != "mix":
        return X_tr, y_tr
    return np.concatenate([X_tr, X_ext]), np.concatenate([y_tr, y_ext])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/muril-base-cased")
    ap.add_argument("--tag", default="muril")
    # data
    ap.add_argument("--max-len", type=int, default=128, help="p99 is 121 tokens")
    ap.add_argument("--no-demojize", action="store_true",
                    help="disable emoji->:name: rewriting (MuRIL needs it; see module docstring)")
    ap.add_argument("--no-dedupe", action="store_true",
                    help="keep repeated comments and label-conflict groups; see "
                         "prep.dedupe_index. Every model sharing SPLIT_SEED must agree on "
                         "this flag or ensemble.py misaligns its OOF rows")
    # external corpus
    ap.add_argument("--external", nargs="?", const=str(EXTERNAL_DEFAULT), default="",
                    metavar="CSV",
                    help="auxiliary corpus with id,Comment,Label[,source_label]; the bare "
                         "flag uses data/external/offenseval_kn.csv")
    ap.add_argument("--external-mode", choices=["stage", "mix"], default="stage",
                    help="stage: fit the external corpus first, then start every fold from "
                         "those weights. mix: concatenate it into each TRAINING fold only. "
                         "stage is the default because the external labels are noisy")
    ap.add_argument("--external-epochs", type=int, default=2,
                    help="stage mode only; 2 is enough to move the encoder without "
                         "letting it memorise the auxiliary label noise")
    ap.add_argument("--external-lr", type=float, default=3e-5)
    ap.add_argument("--external-keep-other", action="store_true",
                    help="keep Offensive_Targeted_Insult_Other rows; they are the noisiest")
    ap.add_argument("--external-reinit-head", action="store_true", default=True)
    ap.add_argument("--no-external-reinit-head", dest="external_reinit_head",
                    action="store_false",
                    help="carry the stage-1 classifier head into stage 2 instead of "
                         "re-initializing it")
    # schedule
    ap.add_argument("--folds", type=int, default=N_SPLITS,
                    help="0 = 15%% holdout, 1 = full-data fit, >1 = OOF folds")
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
    ap.add_argument("--loss", choices=["ce", "focal"], default="ce")
    ap.add_argument("--focal-gamma", type=float, default=2.0,
                    help="only used with --loss focal; 1.0 is mild, 2.0 standard")
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
    ap.add_argument("--no-ema-bias-correct", dest="ema_bias_correct", action="store_false",
                    default=True, help="restore the pre-fix EMA that never divides out "
                                       "the initialization; for reproducing old runs only")
    ap.add_argument("--select", choices=["best", "last"], default="best",
                    help="which checkpoint supplies the fold's predictions. best picks the "
                         "highest macro-F1 on the validation fold, which is the fold being "
                         "reported, so its OOF is optimistic; last is unbiased. Both "
                         "numbers are printed either way")
    ap.add_argument("--rdrop", type=float, default=0.0,
                    help="R-Drop KL weight; 0 disables. Try 0.3-1.0. Doubles cost, "
                         "and stacks with --fgm at ~2.8x baseline")
    ap.add_argument("--aux-weight", type=float, default=0.0,
                    help="weight on an auxiliary head predicting whether the "
                         "comment contains violent language (hastika.task_b.axes). 0 "
                         "disables it. The act axis has 443 positives against "
                         "the Violence class's 221, so this gives the encoder "
                         "twice the signal for the concept Violence depends on")
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
          f"seeds={args.seeds} external={args.external or '-'}"
          f"{'/' + args.external_mode if args.external else ''}", flush=True)

    train = pd.read_csv(RAW_DATA_DIR / "binary_train.csv")
    test = pd.read_csv(RAW_DATA_DIR / "binary_validation_inputs.csv")
    if not args.no_dedupe:
        train = train.iloc[dedupe_index(train["Comment"].tolist(),
                                        train["Label"].tolist(), "task A")].reset_index(drop=True)
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

    X_ext = y_ext = None
    if args.external:
        X_ext, y_ext = load_external(args.external, demoji, not args.external_keep_other)
        print(f"external: {len(y_ext)} rows, {int(y_ext.sum())} Hate, "
              f"from {args.external}", flush=True)

    if X_ext is not None and args.external_mode == "stage":
        # Stage 1. Checkpoint selection uses a slice held out of the EXTERNAL rows,
        # so no HASTIKA row -- and in particular no fold's validation rows -- is
        # seen here. Run once and reused by every seed and fold: repeating it per
        # fold would cost 5x for a stage that never sees the fold split anyway.
        from sklearn.model_selection import train_test_split
        e_tr, e_va = train_test_split(np.arange(len(y_ext)), test_size=0.1,
                                      stratify=y_ext, random_state=SPLIT_SEED)
        stage = copy.copy(args)
        stage.epochs, stage.lr = args.external_epochs, args.external_lr
        stage.head_lr, stage.seed = args.external_lr * 3, args.seeds[0]
        print(f"===== stage 1: external, {args.external_epochs} epochs =====", flush=True)
        # X_test[:1] only feeds the (unused) test loader that train_fold always builds
        f1_ext, _, _, state = train_fold(stage, tok, X_ext[e_tr], y_ext[e_tr],
                                         X_ext[e_va], y_ext[e_va], X_test[:1], device,
                                         "ext", return_state=True)
        print(f"stage 1 held-out macro-F1 {f1_ext:.4f} (external labels, "
              f"not comparable to HASTIKA numbers)", flush=True)
        args.init_state = state

    oof = np.zeros((len(y), 2))
    test_probs = np.zeros((len(X_test), 2))
    n_seeds = len(args.seeds)

    for seed in args.seeds:
        args.seed = seed
        if args.folds and args.folds > 1:
            skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SPLIT_SEED)
            for k, (tr_i, va_i) in enumerate(skf.split(X, y), 1):
                print(f"===== seed {seed} fold {k}/{args.folds} =====", flush=True)
                X_tr, y_tr = with_external(X[tr_i], y[tr_i], X_ext, y_ext, args)
                f1, p_va, p_te = train_fold(args, tok, X_tr, y_tr, X[va_i], y[va_i],
                                            X_test, device, f"s{seed}f{k}")
                oof[va_i] += p_va / n_seeds
                test_probs += p_te / (args.folds * n_seeds)
        elif args.folds == 1:
            X_tr, y_tr = with_external(X, y, X_ext, y_ext, args)
            f1, p_va, p_te = train_fold(args, tok, X_tr, y_tr, np.array([]), np.array([]),
                                         X_test, device, f"s{seed}full")
            test_probs += p_te / n_seeds
        else:
            from sklearn.model_selection import train_test_split
            tr_i, va_i = train_test_split(np.arange(len(y)), test_size=0.15,
                                          stratify=y, random_state=SPLIT_SEED)
            X_tr, y_tr = with_external(X[tr_i], y[tr_i], X_ext, y_ext, args)
            f1, p_va, p_te = train_fold(args, tok, X_tr, y_tr, X[va_i], y[va_i],
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
