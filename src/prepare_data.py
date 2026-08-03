from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="Extract and stratify the waste dataset.")
    p.add_argument("--archive", required=True, help="Path to the dataset zip file")
    p.add_argument("--output", default="data/splits", help="Output split directory")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    archive = Path(args.archive)
    output = Path(args.output)
    temp = output.parent / "raw"
    if output.exists():
        shutil.rmtree(output)
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(temp)

    candidates = [p for p in temp.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not candidates:
        raise RuntimeError("No images found after extraction.")

    labels = [p.parent.name for p in candidates]
    train_paths, test_paths = train_test_split(
        candidates, test_size=0.15, random_state=args.seed, stratify=labels
    )
    train_labels = [p.parent.name for p in train_paths]
    train_paths, val_paths = train_test_split(
        train_paths, test_size=0.1765, random_state=args.seed, stratify=train_labels
    )

    split_map = {"train": train_paths, "val": val_paths, "test": test_paths}
    summary = {}
    for split, paths in split_map.items():
        counts = Counter()
        for src in paths:
            label = src.parent.name
            dst = output / split / label / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            counts[label] += 1
        summary[split] = dict(sorted(counts.items()))

    output.mkdir(parents=True, exist_ok=True)
    (output / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
