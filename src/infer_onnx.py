from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

from src.data import IMAGENET_MEAN, IMAGENET_STD


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/waste_classifier.onnx")
    p.add_argument("--image", required=True)
    p.add_argument("--classes", default="artifacts/classes.json")
    p.add_argument("--image-size", type=int, default=224)
    args = p.parse_args()

    classes = json.loads(Path(args.classes).read_text(encoding="utf-8"))["classes"]
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    image = transform(Image.open(args.image).convert("RGB")).unsqueeze(0).numpy().astype(np.float32)
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    logits = session.run(None, {session.get_inputs()[0].name: image})[0][0]
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    idx = int(probs.argmax())
    print(json.dumps({"prediction": classes[idx], "confidence": float(probs[idx])}, indent=2))


if __name__ == "__main__":
    main()
