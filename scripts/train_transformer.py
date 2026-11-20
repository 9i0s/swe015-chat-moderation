"""Area C - fine-tune DistilBERTurk (dbmdz/distilbert-base-turkish-cased) for offensive / clean.

The pretrained encoder (6 layers, 768 hidden, WordPiece vocab of 32k) gets a classification head
(pre-classifier + dropout + linear layer on the [CLS] position) and is fine-tuned end-to-end with
AdamW and cross-entropy. The checkpoint with the best validation macro-F1 is kept.

Usage:  python scripts/train_transformer.py [--epochs 3] [--base models/distilberturk-base]
"""
import argparse
import json
import math
import pathlib
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import (BertTokenizerFast, DistilBertForSequenceClassification,
                          get_linear_schedule_with_warmup)

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moderation.normalize import light_normalize  # noqa: E402

MAX_LEN = 96


def load(split):
    df = pd.read_csv(ROOT / "data" / f"{split}.tsv", sep="\t", quoting=3, dtype={"id": str})
    return [light_normalize(t) for t in df["text"].astype(str)], df["label"].to_numpy()


def length_bucketed_batches(lengths, batch_size, rng):
    """Shuffle, then sort inside mega-batches so each batch has similar lengths (less padding)."""
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    mega = batch_size * 50
    batches = []
    for i in range(0, len(idx), mega):
        chunk = sorted(idx[i:i + mega], key=lambda j: lengths[j])
        batches += [chunk[k:k + batch_size] for k in range(0, len(chunk), batch_size)]
    rng.shuffle(batches)
    return batches


@torch.no_grad()
def predict_proba(model, tok, texts, batch_size=64):
    model.eval()
    order = np.argsort([len(t) for t in texts])
    probs = np.zeros(len(texts))
    for i in range(0, len(texts), batch_size):
        ids = order[i:i + batch_size]
        enc = tok([texts[j] for j in ids], truncation=True, max_length=MAX_LEN, padding=True,
                  return_tensors="pt")
        logits = model(**enc).logits
        probs[ids] = torch.softmax(logits, -1)[:, 1].numpy()
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "models" / "distilberturk-base"))
    ap.add_argument("--out", default=str(ROOT / "models" / "distilberturk-offense-fp32"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    if not pathlib.Path(args.base).exists():          # fall back to the Hugging Face Hub copy
        args.base = "dbmdz/distilbert-base-turkish-cased"
    tok = BertTokenizerFast.from_pretrained(args.base, do_lower_case=False)
    model = DistilBertForSequenceClassification.from_pretrained(args.base, num_labels=2)

    xtr, ytr = load("train")
    xva, yva = load("val")
    enc_tr = tok(xtr, truncation=True, max_length=MAX_LEN)
    lengths = [len(x) for x in enc_tr["input_ids"]]

    steps_per_epoch = math.ceil(len(xtr) / args.batch)
    total = steps_per_epoch * args.epochs
    no_decay = ["bias", "LayerNorm.weight"]
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=args.lr)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    loss_fn = torch.nn.CrossEntropyLoss()

    log = {"args": vars(args), "steps_per_epoch": steps_per_epoch, "epochs": []}
    best = -1.0
    t0 = time.time()
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for bi, batch in enumerate(length_bucketed_batches(lengths, args.batch, rng), 1):
            feats = tok.pad({"input_ids": [enc_tr["input_ids"][j] for j in batch],
                             "attention_mask": [enc_tr["attention_mask"][j] for j in batch]},
                            return_tensors="pt")
            labels = torch.tensor(ytr[batch], dtype=torch.long)
            logits = model(input_ids=feats["input_ids"], attention_mask=feats["attention_mask"]).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            run_loss += loss.item(); step += 1
            if bi % 50 == 0:
                el = time.time() - t0
                print(f"epoch {epoch} step {bi}/{steps_per_epoch} loss {run_loss / bi:.4f} "
                      f"elapsed {el / 60:.1f} min, eta {(total - step) * el / step / 60:.1f} min", flush=True)
        pva = predict_proba(model, tok, xva)
        f1 = f1_score(yva, (pva >= 0.5).astype(int), average="macro")
        rec = {"epoch": epoch, "train_loss": run_loss / steps_per_epoch, "val_macro_f1@0.5": round(f1, 4),
               "minutes": round((time.time() - t0) / 60, 1)}
        log["epochs"].append(rec)
        print(json.dumps(rec), flush=True)
        if f1 > best:
            best = f1
            model.save_pretrained(args.out)
            tok.save_pretrained(args.out)
            print(f"  saved best checkpoint (val macro-F1 {f1:.4f})", flush=True)
    log["best_val_macro_f1@0.5"] = round(best, 4)
    log["train_minutes"] = round((time.time() - t0) / 60, 1)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "transformer_training.json").write_text(json.dumps(log, indent=2))
    print("done", json.dumps(log))


if __name__ == "__main__":
    main()
