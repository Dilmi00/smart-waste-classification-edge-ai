from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.model import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/best_model.pt")
    p.add_argument("--output", default="models/waste_classifier.onnx")
    args = p.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    classes = checkpoint["classes"]
    image_size = checkpoint.get("image_size", 224)
    model = build_model(len(classes), pretrained=False, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dummy = torch.randn(1, 3, image_size, image_size)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
    )
    print(f"Exported ONNX model to {args.output}")


if __name__ == "__main__":
    main()
