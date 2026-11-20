"""Single run command:  python run_app.py   -> open http://127.0.0.1:8000

Downloads the trained model weights from Google Drive on first start if they are missing
(see scripts/download_models.py), then starts the FastAPI server.
"""
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
NEEDED = [ROOT / "models" / "distilberturk-offense-int8" / "model_int8.pt",
          ROOT / "models" / "distilberturk-offense-fp32" / "model.safetensors",
          ROOT / "models" / "tfidf_lr.joblib",
          ROOT / "models" / "w2v_ffnn" / "ffnn.pt"]

if __name__ == "__main__":
    if not all(p.exists() for p in NEEDED):
        print("Model weights not found - downloading them (one time)...")
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "download_models.py")])
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    print(f"GuardChat running on http://127.0.0.1:{port}")
    uvicorn.run("app.server:app", host="127.0.0.1", port=port, log_level="warning")
