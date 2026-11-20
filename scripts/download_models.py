"""Download the trained model weights (not stored in git) from Google Drive and unpack them into models/.

  part 1: INT8 DistilBERTurk + tf-idf/LR + word2vec/FFNN   (everything the app needs by default)
  part 2: FP32 DistilBERTurk (for the side-by-side comparison and for re-running the evaluation)

Usage:  python scripts/download_models.py
"""
import pathlib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = {
    "guardchat_models_part1.zip": "GDRIVE_ID_PART1",
    "guardchat_models_part2_fp32.zip": "GDRIVE_ID_PART2",
}


def main():
    import gdown
    for name, file_id in FILES.items():
        dst = ROOT / name
        if not dst.exists():
            print(f"Downloading {name} ...")
            gdown.download(id=file_id, output=str(dst), quiet=False)
        with zipfile.ZipFile(dst) as z:
            z.extractall(ROOT)
        print(f"Unpacked {name}")
    print("All model files are in", ROOT / "models")


if __name__ == "__main__":
    main()
