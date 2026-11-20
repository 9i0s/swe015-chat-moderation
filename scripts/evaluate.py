"""Evaluate every model of the ladder on the SAME held-out test set.

* decision threshold per model tuned on the validation split (macro-F1), then frozen for test
* metrics: macro-F1 (main), accuracy, offensive-class precision/recall/F1, 95% bootstrap CI
* size: bytes on disk of everything the model needs at inference time
* latency: end-to-end (normalisation + features + model) for ONE message, batch size 1,
  1 CPU thread, mean / p50 / p95 over 200 test messages after 10 warm-up runs
Usage:  python scripts/evaluate.py
"""
import json
import pathlib
import platform
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moderation.models import load_all  # noqa: E402

LATENCY_BUDGET_MS = 25.0      # set before measuring: 1 CPU core must sustain >= 40 messages / second
N_LATENCY = 200
ORDER = ["lexicon", "tfidf", "w2v_ffnn", "bert_fp32", "bert_int8"]


def load(split):
    return pd.read_csv(ROOT / "data" / f"{split}.tsv", sep="\t", quoting=3, dtype={"id": str})


def tune_threshold(y, p):
    grid = np.round(np.arange(0.02, 0.99, 0.01), 2)
    scores = [f1_score(y, (p >= t).astype(int), average="macro") for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def bootstrap_ci(y, pred, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        vals.append(f1_score(y[idx], pred[idx], average="macro"))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def cpu_name():
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        return next(line.split(":", 1)[1].strip() for line in out.splitlines() if line.startswith("Model name"))
    except Exception:
        return platform.processor()


def main():
    torch.set_num_threads(1)
    val, test = load("val"), load("test")
    xva, yva = val["text"].astype(str).tolist(), val["label"].to_numpy()
    xte, yte = test["text"].astype(str).tolist(), test["label"].to_numpy()
    models = load_all(ORDER)

    rng = random.Random(0)
    lat_msgs = rng.sample(xte, N_LATENCY)
    results, thresholds, probs_test = [], {}, {}
    for key in ORDER:
        m = models[key]
        t0 = time.time()
        pva = m.predict_proba(xva)
        pte = m.predict_proba(xte)
        thr, f1_val = tune_threshold(yva, pva)
        thresholds[key] = thr
        pred = (pte >= thr).astype(int)
        probs_test[key] = pte
        p, r, f, _ = precision_recall_fscore_support(yte, pred, labels=[0, 1], zero_division=0)
        lo, hi = bootstrap_ci(yte, pred)
        # latency, one message at a time (lexicon: start from an empty token cache)
        if key == "lexicon":
            m.m.token_score.cache_clear()
        for msg in lat_msgs[:10]:
            m.predict_proba([msg])
        lat = []
        for msg in lat_msgs:
            s = time.perf_counter()
            m.predict_proba([msg])
            lat.append((time.perf_counter() - s) * 1000)
        lat = np.array(lat)
        row = {"key": key, "model": m.name, "area": m.area, "precision": m.precision,
               "threshold": thr, "val_macro_f1": round(f1_val, 4),
               "test_macro_f1": round(f1_score(yte, pred, average="macro"), 4),
               "test_macro_f1_ci95": [round(lo, 4), round(hi, 4)],
               "test_macro_f1@0.5": round(f1_score(yte, (pte >= 0.5).astype(int), average="macro"), 4),
               "test_accuracy": round(accuracy_score(yte, pred), 4),
               "offensive_precision": round(float(p[1]), 4), "offensive_recall": round(float(r[1]), 4),
               "offensive_f1": round(float(f[1]), 4), "clean_f1": round(float(f[0]), 4),
               "confusion_matrix[[TN,FP],[FN,TP]]": confusion_matrix(yte, pred).tolist(),
               "size_mb": round(m.size_mb(), 1) if m.size_mb() >= 1 else round(m.size_mb(), 3),
               "latency_ms_mean": round(float(lat.mean()), 2), "latency_ms_p50": round(float(np.median(lat)), 2),
               "latency_ms_p95": round(float(np.percentile(lat, 95)), 2),
               "meets_budget": bool(lat.mean() <= LATENCY_BUDGET_MS),
               "eval_seconds": round(time.time() - t0, 1)}
        results.append(row)
        print(json.dumps(row), flush=True)

    # ablation on the validation split: Transformer with / without inference-time de-obfuscation
    ablation = {}
    for key in ["bert_fp32", "bert_int8"]:
        m = models[key]
        with_d = tune_threshold(yva, m.predict_proba(xva))[1]
        m.use_deobfuscate = False
        without_d = tune_threshold(yva, m.predict_proba(xva))[1]
        m.use_deobfuscate = True
        ablation[key] = {"val_macro_f1_with_deobfuscate": round(with_d, 4),
                         "val_macro_f1_without": round(without_d, 4)}
    print("de-obfuscation ablation (val):", ablation, flush=True)

    fp, i8 = probs_test["bert_fp32"], probs_test["bert_int8"]
    agree = float(np.mean((fp >= thresholds["bert_fp32"]) == (i8 >= thresholds["bert_int8"])))
    hardware = {"cpu": cpu_name(), "threads": 1, "batch_size": 1, "torch": torch.__version__,
                "python": platform.python_version(), "os": platform.platform(),
                "latency_runs_per_model": N_LATENCY, "warmup_runs": 10}
    summary = {"test_set": {"n": len(yte), "offensive": int(yte.sum())}, "latency_budget_ms": LATENCY_BUDGET_MS,
               "hardware": hardware, "fp32_int8_label_agreement": round(agree, 4),
               "fp32_int8_mean_abs_prob_diff": round(float(np.mean(np.abs(fp - i8))), 4),
               "deobfuscation_ablation_val": ablation, "models": results}
    (ROOT / "results" / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (ROOT / "models" / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

    # per-message predictions (for error analysis and failure examples)
    out = test[["id", "text", "label"]].copy()
    for key in ORDER:
        out[f"p_{key}"] = np.round(probs_test[key], 4)
        out[f"pred_{key}"] = (probs_test[key] >= thresholds[key]).astype(int)
    out.to_csv(ROOT / "results" / "test_predictions.tsv", sep="\t", index=False)

    # markdown table
    lines = [f"Test set: {len(yte)} tweets ({int(yte.sum())} offensive). Hardware: {hardware['cpu']}, "
             f"1 thread, batch 1, {N_LATENCY} runs. Latency budget: {LATENCY_BUDGET_MS:.0f} ms/message.", "",
             "| Model | Area | Number format | Macro-F1 (95% CI) | Off. recall | Size (MB) | Latency mean / p95 (ms) | Budget |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['model']} | {r['area']} | {r['precision']} | {r['test_macro_f1']:.3f} "
                     f"({r['test_macro_f1_ci95'][0]:.3f}-{r['test_macro_f1_ci95'][1]:.3f}) | "
                     f"{r['offensive_recall']:.3f} | {r['size_mb']:g} | {r['latency_ms_mean']:.1f} / "
                     f"{r['latency_ms_p95']:.1f} | {'yes' if r['meets_budget'] else 'no'} |")
    lines.append(f"\nFP32 vs INT8 Transformer: identical labels on {agree * 100:.1f}% of test messages.")
    (ROOT / "results" / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
