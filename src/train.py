from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.common import save_json, set_seed
from src.data import make_loaders
from src.model import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MobileNetV3-Small with MLflow tracking."
    )
    parser.add_argument("--data", default="data/splits")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment", default="waste-classification")
    parser.add_argument(
        "--unfreeze",
        action="store_true",
        help="Fine-tune the full backbone instead of only the classifier.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not use ImageNet-pretrained weights.",
    )
    return parser.parse_args()


def evaluate(model, loader, device):
    model.eval()
    losses = []
    y_true = []
    y_pred = []

    criterion = nn.CrossEntropyLoss()

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            losses.append(loss.item())
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loaders, classes = make_loaders(
        args.data,
        args.batch_size,
        args.image_size,
        args.workers,
    )

    model = build_model(
        num_classes=len(classes),
        pretrained=not args.no_pretrained,
        freeze_backbone=not args.unfreeze,
    ).to(device)

    train_labels = [label for _, label in loaders["train"].dataset.samples]
    train_counts = np.bincount(train_labels, minlength=len(classes))

    if np.any(train_counts == 0):
        missing = [
            classes[index]
            for index, count in enumerate(train_counts)
            if count == 0
        ]
        raise ValueError(
            f"The training split contains no images for these classes: {missing}"
        )

    class_weights = train_counts.sum() / (len(classes) * train_counts)
    class_weights_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.lr,
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment(args.experiment)

    models_dir = Path("models")
    artifacts_dir = Path("artifacts")
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    best_path = models_dir / "best_model.pt"

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "model": "mobilenet_v3_small",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "image_size": args.image_size,
                "pretrained": not args.no_pretrained,
                "freeze_backbone": not args.unfreeze,
                "seed": args.seed,
                "device": str(device),
                "num_classes": len(classes),
                "classes": ",".join(classes),
            }
        )

        best_f1 = -1.0
        best_epoch = 0
        best_val_accuracy = 0.0
        started = time.perf_counter()

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0

            for images, labels in loaders["train"]:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad(set_to_none=True)

                logits = model(images)
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss / max(1, len(loaders["train"]))
            validation = evaluate(model, loaders["val"], device)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": validation["loss"],
                    "val_accuracy": validation["accuracy"],
                    "val_f1_macro": validation["f1_macro"],
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_acc={validation['accuracy']:.4f} | "
                f"val_f1={validation['f1_macro']:.4f}"
            )

            if validation["f1_macro"] > best_f1:
                best_f1 = validation["f1_macro"]
                best_epoch = epoch
                best_val_accuracy = validation["accuracy"]

                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "classes": classes,
                        "image_size": args.image_size,
                        "architecture": "mobilenet_v3_small",
                        "best_epoch": best_epoch,
                        "best_val_accuracy": best_val_accuracy,
                        "best_val_f1_macro": best_f1,
                    },
                    best_path,
                )

        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])

        test_results = evaluate(model, loaders["test"], device)
        elapsed = time.perf_counter() - started

        mlflow.log_metrics(
            {
                "best_epoch": float(best_epoch),
                "best_val_accuracy": float(best_val_accuracy),
                "best_val_f1_macro": float(best_f1),
                "test_accuracy": test_results["accuracy"],
                "test_f1_macro": test_results["f1_macro"],
                "training_seconds": elapsed,
            }
        )

        report = classification_report(
            test_results["y_true"],
            test_results["y_pred"],
            target_names=classes,
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(
            test_results["y_true"],
            test_results["y_pred"],
        ).tolist()

        save_json(report, artifacts_dir / "classification_report.json")
        save_json(
            {"classes": classes, "matrix": matrix},
            artifacts_dir / "confusion_matrix.json",
        )
        save_json(
            {
                "classes": classes,
                "class_counts": train_counts.tolist(),
                "class_weights": class_weights.tolist(),
            },
            artifacts_dir / "dataset_summary.json",
        )

        # Log generated reports and the best PyTorch checkpoint.
        # We intentionally log the checkpoint as a normal artifact instead of
        # mlflow.pytorch.log_model(), avoiding the MLflow PT2 input-example error.
        mlflow.log_artifacts(str(artifacts_dir), artifact_path="evaluation")
        mlflow.log_artifact(str(best_path), artifact_path="model_checkpoint")

        summary = {
            "run_id": run.info.run_id,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_accuracy,
            "best_val_f1_macro": best_f1,
            "test_accuracy": test_results["accuracy"],
            "test_f1_macro": test_results["f1_macro"],
            "training_seconds": elapsed,
            "checkpoint": str(best_path),
        }

        print("\nTraining completed successfully.")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()