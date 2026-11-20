"""Area D - post-training dynamic INT8 quantization of the fine-tuned DistilBERTurk.

All nn.Linear layers (Q/K/V/output projections, feed-forward layers, classifier) get INT8 weights;
activations are quantized on the fly per batch (dynamic quantization, fbgemm/x86 kernels on CPU).
Embeddings and LayerNorm stay FP32. No calibration data or retraining is needed.

Usage:  python scripts/quantize.py
"""
import json
import pathlib
import shutil

import torch
from torch import nn
from transformers import DistilBertForSequenceClassification

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "models" / "distilberturk-offense-fp32"
DST = ROOT / "models" / "distilberturk-offense-int8"


def main():
    model = DistilBertForSequenceClassification.from_pretrained(SRC).eval()
    qmodel = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    DST.mkdir(parents=True, exist_ok=True)
    torch.save(qmodel.state_dict(), DST / "model_int8.pt")
    for f in ["config.json", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json"]:
        if (SRC / f).exists():
            shutil.copy(SRC / f, DST / f)
    n_linear = sum(isinstance(m, torch.ao.nn.quantized.dynamic.Linear) for m in qmodel.modules())
    info = {"method": "torch.ao.quantization.quantize_dynamic, qint8, nn.Linear only",
            "quantized_linear_layers": n_linear,
            "fp32_weights_mb": round((SRC / "model.safetensors").stat().st_size / 1e6, 1),
            "int8_weights_mb": round((DST / "model_int8.pt").stat().st_size / 1e6, 1)}
    (ROOT / "results" / "quantization.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
