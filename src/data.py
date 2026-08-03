from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 224):
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tfms, eval_tfms


def load_split_datasets(split_root: str | Path, image_size: int = 224):
    split_root = Path(split_root)
    train_tfms, eval_tfms = build_transforms(image_size)
    train_ds = datasets.ImageFolder(split_root / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(split_root / "val", transform=eval_tfms)
    test_ds = datasets.ImageFolder(split_root / "test", transform=eval_tfms)
    if train_ds.classes != val_ds.classes or train_ds.classes != test_ds.classes:
        raise ValueError("Class ordering differs across dataset splits.")
    return train_ds, val_ds, test_ds


def make_loaders(split_root: str | Path, batch_size: int, image_size: int, workers: int = 0):
    train_ds, val_ds, test_ds = load_split_datasets(split_root, image_size)
    pin = torch.cuda.is_available()
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin),
    }
    return loaders, train_ds.classes
