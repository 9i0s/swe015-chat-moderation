"""Area A - our own word2vec (skip-gram) trained on the training messages, feeding a feedforward
network trained with mini-batch SGD (lr 0.01, momentum 0.9) and backpropagation.

Message vector = [mean ; max] of the word2vec vectors of its tokens (200 dims).
Network: 200 -> 128 (ReLU, dropout 0.3) -> 64 (ReLU, dropout 0.3) -> 2, class-weighted cross-entropy,
early stopping on validation macro-F1.
Usage:  python scripts/train_w2v_ffnn.py
"""
import json
import pathlib
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from sklearn.metrics import f1_score
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moderation.models import FFNN, w2v_features  # noqa: E402
from moderation.normalize import normalize  # noqa: E402

SEED = 42
HIDDEN, DROPOUT = (128, 64), 0.3


def load(split):
    df = pd.read_csv(ROOT / "data" / f"{split}.tsv", sep="\t", quoting=3, dtype={"id": str})
    return df["text"].astype(str).tolist(), df["label"].to_numpy()


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(1)
    out = ROOT / "models" / "w2v_ffnn"
    out.mkdir(parents=True, exist_ok=True)
    xtr, ytr = load("train")
    xva, yva = load("val")

    # 1) word2vec (skip-gram with negative sampling) on the training messages only
    t0 = time.time()
    sents = [normalize(t).split() for t in xtr]
    w2v = Word2Vec(sentences=sents, vector_size=100, window=5, min_count=2, sg=1, negative=10,
                   epochs=15, workers=1, seed=SEED)
    w2v.wv.save(str(out / "word2vec.kv"))
    w2v_sec = time.time() - t0
    print(f"word2vec: vocab {len(w2v.wv)}, {w2v_sec:.0f}s", flush=True)
    for probe in ["salak", "amk", "maç", "güzel"]:
        if probe in w2v.wv:
            print(f"  nearest to {probe!r}:", [w for w, _ in w2v.wv.most_similar(probe, topn=5)], flush=True)

    # 2) feedforward network trained with SGD + backpropagation
    Xtr = torch.tensor(w2v_features(w2v.wv, xtr)); Ytr = torch.from_numpy(ytr).long()
    Xva = torch.tensor(w2v_features(w2v.wv, xva))
    net = FFNN(Xtr.shape[1], HIDDEN, DROPOUT)
    net.mu.copy_(Xtr.mean(0)); net.sigma.copy_(Xtr.std(0) + 1e-6)     # standardise inputs
    pos = float(ytr.mean())
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1 / (1 - pos), 1 / pos]) / 2)
    opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    best, best_state, patience, history = -1, None, 0, []
    for epoch in range(1, 101):
        net.train()
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            loss = loss_fn(net(Xtr[idx]), Ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()          # backpropagation + SGD step
            tot += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            pva = torch.softmax(net(Xva), -1)[:, 1].numpy()
        f1 = f1_score(yva, (pva >= 0.5).astype(int), average="macro")
        history.append({"epoch": epoch, "train_loss": round(tot / len(Xtr), 4), "val_macro_f1@0.5": round(f1, 4)})
        if f1 > best:
            best, patience = f1, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
        if epoch % 5 == 0:
            print(history[-1], flush=True)
        if patience >= 15:
            break
    torch.save(best_state, out / "ffnn.pt")
    meta = {"hidden": list(HIDDEN), "dropout": DROPOUT, "w2v": {"dim": 100, "window": 5, "sg": 1,
            "min_count": 2, "epochs": 15, "vocab": len(w2v.wv)}, "optimizer": "SGD lr=0.01 momentum=0.9 wd=1e-4",
            "batch": 64, "best_val_macro_f1@0.5": round(best, 4), "epochs_run": len(history),
            "word2vec_seconds": round(w2v_sec, 1), "total_seconds": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (ROOT / "results" / "w2v_ffnn_training.json").write_text(json.dumps({**meta, "history": history}, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
