"""Area B - classical baseline: tf-idf (word 1-2 grams + character 2-5 grams) + logistic regression.

The regularisation strength C is chosen on the validation split by macro-F1.
Usage:  python scripts/train_tfidf.py
"""
import json
import pathlib
import sys
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moderation.normalize import normalize  # noqa: E402


def load(split):
    df = pd.read_csv(ROOT / "data" / f"{split}.tsv", sep="\t", quoting=3, dtype={"id": str})
    return [normalize(t) for t in df["text"].astype(str)], df["label"].to_numpy()


def build(C):
    feats = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2), min_df=2,
                                 sublinear_tf=True, max_features=100_000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True,
                                 max_features=200_000)),
    ])
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, solver="liblinear")
    return Pipeline([("tfidf", feats), ("lr", clf)])


def main():
    xtr, ytr = load("train")
    xva, yva = load("val")
    t0 = time.time()
    search = {}
    best_c, best_f1 = None, -1
    for C in [0.5, 1, 2, 4, 8]:
        pipe = build(C).fit(xtr, ytr)
        f1 = f1_score(yva, pipe.predict(xva), average="macro")
        search[str(C)] = round(f1, 4)
        print(f"C={C}: val macro-F1 {f1:.4f}", flush=True)
        if f1 > best_f1:
            best_c, best_f1 = C, f1
    pipe = build(best_c).fit(xtr, ytr)
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(pipe, ROOT / "models" / "tfidf_lr.joblib", compress=3)
    info = {"C_search_val_macro_f1": search, "best_C": best_c, "best_val_macro_f1": round(best_f1, 4),
            "n_features": len(pipe.named_steps["tfidf"].transformer_list[0][1].vocabulary_)
            + len(pipe.named_steps["tfidf"].transformer_list[1][1].vocabulary_),
            "train_seconds": round(time.time() - t0, 1)}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "tfidf_training.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
