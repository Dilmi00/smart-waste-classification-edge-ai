from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from src.common import save_json
from src.model import build_model


def avg_latency(fn, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        fn()
    started = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - started) * 1000 / runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/best_model.pt")
    p.add_argument("--onnx", default="models/waste_classifier.onnx")
    p.add_argument("--runs", type=int, default=100)
    args = p.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    classes = checkpoint["classes"]
    image_size = checkpoint.get("image_size", 224)
    model = build_model(len(classes), pretrained=False, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    x = torch.randn(1, 3, image_size, image_size)

    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    x_np = x.numpy().astype(np.float32)

    with torch.inference_mode():
        pytorch_ms = avg_latency(lambda: model(x), 10, args.runs)
    onnx_ms = avg_latency(lambda: session.run(None, {input_name: x_np}), 10, args.runs)

    with torch.inference_mode():
        pt_logits = model(x).numpy()
    onnx_logits = session.run(None, {input_name: x_np})[0]
    agreement = int(pt_logits.argmax(1)[0] == onnx_logits.argmax(1)[0])

    result = {
        "pytorch_latency_ms": pytorch_ms,
        "onnx_latency_ms": onnx_ms,
        "speedup": pytorch_ms / onnx_ms if onnx_ms else None,
        "prediction_agreement": bool(agreement),
        "checkpoint_size_mb": Path(args.checkpoint).stat().st_size / 1_048_576,
        "onnx_size_mb": Path(args.onnx).stat().st_size / 1_048_576,
        "runs": args.runs,
    }
    save_json(result, "artifacts/benchmark.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
