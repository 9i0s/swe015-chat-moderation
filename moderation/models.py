"""Common interface for the five models of the ladder (used by evaluation and by the app).

Every model takes raw chat messages and returns P(offensive); its own preprocessing is included,
so measured latency is end-to-end (normalisation + features + model).
"""
import json
import pathlib

import numpy as np
import torch
from torch import nn

from .lexicon import LexiconMatcher
from .normalize import deobfuscate, light_normalize, normalize

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def dir_size_mb(*paths) -> float:
    total = 0
    for p in paths:
        p = pathlib.Path(p)
        files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
        total += sum(f.stat().st_size for f in files)
    return total / 1e6


class LexiconModel:
    key, name, area, precision = "lexicon", "Edit-distance lexicon (rules)", "B", "-"

    def __init__(self):
        self.m = LexiconMatcher()

    def predict_proba(self, texts):
        return np.array([self.m.score(t) for t in texts])

    def size_mb(self):
        return dir_size_mb(ROOT / "moderation" / "lexicon.py")


class TfidfLRModel:
    key, name, area, precision = "tfidf", "tf-idf + logistic regression", "B", "FP64"
    path = MODELS / "tfidf_lr.joblib"

    def __init__(self):
        import joblib
        self.pipe = joblib.load(self.path)

    def predict_proba(self, texts):
        return self.pipe.predict_proba([normalize(t) for t in texts])[:, 1]

    def size_mb(self):
        return dir_size_mb(self.path)


class FFNN(nn.Module):
    """Feedforward network on [mean ; max] pooled word2vec vectors."""

    def __init__(self, dim_in, hidden=(128, 64), dropout=0.3):
        super().__init__()
        layers, d = [], dim_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)
        # feature standardisation (statistics of the training split), stored with the weights
        self.register_buffer("mu", torch.zeros(dim_in))
        self.register_buffer("sigma", torch.ones(dim_in))

    def forward(self, x):
        return self.net((x - self.mu) / self.sigma)


def w2v_features(kv, texts):
    dim = kv.vector_size
    out = np.zeros((len(texts), 2 * dim), dtype=np.float32)
    for i, t in enumerate(texts):
        vecs = [kv[w] for w in normalize(t).split() if w in kv.key_to_index]
        if vecs:
            v = np.stack(vecs)
            out[i, :dim] = v.mean(0)
            out[i, dim:] = v.max(0)
    return out


class W2VFFNNModel:
    key, name, area, precision = "w2v_ffnn", "word2vec + feedforward NN", "A", "FP32"
    dir = MODELS / "w2v_ffnn"

    def __init__(self):
        from gensim.models import KeyedVectors
        self.kv = KeyedVectors.load(str(self.dir / "word2vec.kv"))
        meta = json.loads((self.dir / "meta.json").read_text())
        self.net = FFNN(2 * self.kv.vector_size, tuple(meta["hidden"]), meta["dropout"])
        self.net.load_state_dict(torch.load(self.dir / "ffnn.pt", weights_only=True))
        self.net.eval()

    @torch.no_grad()
    def predict_proba(self, texts):
        x = torch.from_numpy(w2v_features(self.kv, texts))
        return torch.softmax(self.net(x), -1)[:, 1].numpy()

    def size_mb(self):
        return dir_size_mb(self.dir)


class TransformerModel:
    """Fine-tuned DistilBERTurk, FP32 or dynamically quantized INT8.

    Input pipeline: light_normalize (as in training) + deobfuscate (inference only: rewrites
    obfuscated words such as "s1kt1r" or "a.m.k"). On validation it changes 9 of 3,176 tweets and leaves
    macro-F1 unchanged (FP32 0.787, INT8 0.784 with and without it); we keep it because deliberate
    obfuscation is far more common in game chat than in tweets.
    """
    MAX_LEN = 96

    def __init__(self, int8: bool = False, use_deobfuscate: bool = True):
        self.use_deobfuscate = use_deobfuscate
        from transformers import BertTokenizerFast, DistilBertConfig, DistilBertForSequenceClassification
        self.int8 = int8
        if int8:
            self.dir = MODELS / "distilberturk-offense-int8"
            self.key, self.name, self.precision = "bert_int8", "DistilBERTurk fine-tuned (INT8)", "INT8 (dynamic)"
            cfg = DistilBertConfig.from_pretrained(self.dir)
            model = DistilBertForSequenceClassification(cfg)
            model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
            model.load_state_dict(torch.load(self.dir / "model_int8.pt", weights_only=False))
        else:
            self.dir = MODELS / "distilberturk-offense-fp32"
            self.key, self.name, self.precision = "bert_fp32", "DistilBERTurk fine-tuned (FP32)", "FP32"
            model = DistilBertForSequenceClassification.from_pretrained(self.dir)
        self.area = "C+D" if int8 else "C"
        self.model = model.eval()
        self.tok = BertTokenizerFast.from_pretrained(self.dir)

    @torch.no_grad()
    def predict_proba(self, texts, batch_size=None):
        # Dynamic INT8 quantization computes the activation scale per batch (padding included), so a
        # message's score depends on its batch-mates. The app scores one message at a time, so the INT8
        # model is always run with batch size 1. FP32 results do not depend on batching (checked), so
        # FP32 uses batches of 64 for speed.
        if batch_size is None:
            batch_size = 1 if self.int8 else 64
        texts = [light_normalize(t) for t in texts]
        if self.use_deobfuscate:
            texts = [deobfuscate(t) for t in texts]
        out = np.zeros(len(texts))
        order = np.argsort([len(t) for t in texts])
        for i in range(0, len(texts), batch_size):
            ids = order[i:i + batch_size]
            enc = self.tok([texts[j] for j in ids], truncation=True, max_length=self.MAX_LEN,
                           padding=True, return_tensors="pt")
            out[ids] = torch.softmax(self.model(**enc).logits, -1)[:, 1].numpy()
        return out

    def size_mb(self):
        weights = "model_int8.pt" if self.int8 else "model.safetensors"
        return dir_size_mb(self.dir / weights)


def load_all(include=("lexicon", "tfidf", "w2v_ffnn", "bert_fp32", "bert_int8")):
    makers = {"lexicon": LexiconModel, "tfidf": TfidfLRModel, "w2v_ffnn": W2VFFNNModel,
              "bert_fp32": lambda: TransformerModel(False), "bert_int8": lambda: TransformerModel(True)}
    return {k: makers[k]() for k in include}


def load_thresholds():
    p = MODELS / "thresholds.json"
    return json.loads(p.read_text()) if p.exists() else {}
