"""GuardChat - real-time Turkish chat moderation (FastAPI back end + static web front end).

Run:  python run_app.py          (or: uvicorn app.server:app --port 8000)
The default model is the INT8-quantized DistilBERTurk; every other model of the ladder can be
selected live, or all five can be compared side by side.
"""
import pathlib
import sys
import time

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moderation.lexicon import LexiconMatcher  # noqa: E402
from moderation.models import load_all, load_thresholds  # noqa: E402
from moderation.normalize import light_normalize, normalize  # noqa: E402

torch.set_num_threads(1)          # CPU-only server: one core per request, as in the benchmark
DEFAULT_MODEL = "bert_int8"
BUDGET_MS = 25.0
ORDER = ["bert_int8", "bert_fp32", "w2v_ffnn", "tfidf", "lexicon"]

MODELS = load_all(ORDER)
THRESHOLDS = load_thresholds()
LEXICON = LexiconMatcher()
for m in MODELS.values():                     # warm-up (first call allocates buffers)
    m.predict_proba(["merhaba nasılsın"])

app = FastAPI(title="GuardChat - Turkish chat moderation")
STATIC = pathlib.Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class Message(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    model: str = DEFAULT_MODEL
    compare: bool = False


def run_model(key: str, text: str) -> dict:
    m = MODELS[key]
    t = time.perf_counter()
    p = float(m.predict_proba([text])[0])
    ms = (time.perf_counter() - t) * 1000
    thr = THRESHOLDS.get(key, 0.5)
    off = p >= thr
    # confidence = how far P(offensive) is from the model's tuned threshold, rescaled to 50-100 %
    conf = 0.5 + 0.5 * ((p - thr) / (1 - thr) if off else (thr - p) / thr)
    return {"key": key, "name": m.name, "precision": m.precision, "prob_offensive": round(p, 4),
            "threshold": thr, "label": "offensive" if off else "clean",
            "confidence": round(conf, 4), "latency_ms": round(ms, 2)}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/models")
def models():
    return {"default": DEFAULT_MODEL, "budget_ms": BUDGET_MS,
            "models": [{"key": k, "name": MODELS[k].name, "precision": MODELS[k].precision,
                        "area": MODELS[k].area, "size_mb": round(MODELS[k].size_mb(), 1),
                        "threshold": THRESHOLDS.get(k, 0.5)} for k in ORDER]}


@app.post("/api/moderate")
def moderate(msg: Message):
    if msg.model not in MODELS:
        raise HTTPException(400, f"unknown model {msg.model}")
    res = run_model(msg.model, msg.text)
    if res["label"] == "offensive":
        masked = LEXICON.mask(msg.text)
        res["mask_mode"] = "words" if masked != msg.text else "full"
        res["masked"] = masked if masked != msg.text else None
    else:
        res["mask_mode"], res["masked"] = None, None
    res["lexicon_hits"] = [{"token": t, "entry": e, "score": round(s, 2)} for t, e, s in LEXICON.matches(msg.text)]
    res["normalized"] = {"transformer": light_normalize(msg.text), "classical": normalize(msg.text)}
    res["within_budget"] = res["latency_ms"] <= BUDGET_MS
    if msg.compare:
        res["all"] = [res if k == msg.model else run_model(k, msg.text) for k in ORDER]
        res["all"] = [{k: v for k, v in r.items() if k != "all"} for r in res["all"]]
    return res
