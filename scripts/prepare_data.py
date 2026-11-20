"""Download OffensEval-TR 2020 (Turkish, sub-task A) and create a fixed stratified split.

Source: Coltekin (2020), "A Corpus of Turkish Offensive Language on Social Media", LREC 2020,
released for SemEval-2020 Task 12 (OffensEval 2020). The official test labels are not part of the
public mirror we use, so we hold out our own stratified 10 % test set from the labelled release
(31,756 tweets). The same test set is used for every model in the ladder.

Usage:  python scripts/prepare_data.py
Output: data/train.tsv, data/val.tsv, data/test.tsv (id, text, label) + data/split_stats.json
"""
import json
import pathlib
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

# Pinned commit of a public GitHub mirror of the official OffensEval-TR training release.
MIRROR_COMMIT = "30b30f821ed33c5a7e9c73e3c76d882d70904845"
URL = ("https://raw.githubusercontent.com/jb-1811/NLP_with_NN_ws1920_OffensEval2020/"
       f"{MIRROR_COMMIT}/xlmr/data/turkish/full.tsv")
SEED = 42
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main():
    DATA.mkdir(exist_ok=True)
    raw = DATA / "offenseval_tr_2020_full.tsv"
    if not raw.exists():
        print("Downloading", URL)
        urllib.request.urlretrieve(URL, raw)
    df = pd.read_csv(raw, sep="\t", quoting=3, dtype={"id": str})
    df = df.rename(columns={"tweet": "text", "subtask_a": "label"})[["id", "text", "label"]]
    df = df.dropna(subset=["text"]).drop_duplicates(subset=["id"])
    df["label"] = df["label"].astype(int)          # 0 = clean (NOT), 1 = offensive (OFF)

    ids_file = DATA / "split_ids.json"
    if ids_file.exists():                       # reproduce the exact published split
        ids = json.loads(ids_file.read_text())
        by_id = df.set_index("id")
        train, val, test = (by_id.loc[ids[k]].reset_index() for k in ("train", "val", "test"))
    else:
        train, rest = train_test_split(df, test_size=0.20, stratify=df["label"], random_state=SEED)
        val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=SEED)
        ids_file.write_text(json.dumps({"train": train["id"].tolist(), "val": val["id"].tolist(),
                                        "test": test["id"].tolist()}))

    stats = {"source": URL, "seed": SEED, "total": len(df)}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(DATA / f"{name}.tsv", sep="\t", index=False)
        stats[name] = {"n": len(part), "offensive": int(part["label"].sum()),
                       "offensive_ratio": round(float(part["label"].mean()), 4)}
    (DATA / "split_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
