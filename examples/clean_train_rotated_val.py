"""Worked example: train on un-rotated shapes, validate on rotated ones.

This is the same rotation distribution-shift study as
``configs/train_clean_eval_rotated.yaml``, but done by hand against the dataset
API — for when you want full control over the loop rather than a declarative
config. The point: the train and val *distributions* are independent objects,
so a "study" is just two ShapeDatasets that differ in one knob.

    python -m examples.clean_train_rotated_val --epochs 50

Keep --epochs small for a quick smoke; bump it for a real measurement.
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.annotations import BackgroundType
from data.synthetic_shapes_dataset import ShapeDataset
from models.center_heatmap_net import CenterHeatmapNet
from scripts.run_training import evaluate_heatmap
from utils.heatmap_loss import HeatmapLoss


def make_loader(*, rotate: bool, num_images: int, seed: int, image_size, batch_size):
    """One ShapeDataset, varying only rotate_shapes. Everything else is held
    fixed so the train/val difference is rotation and nothing else."""
    ds = ShapeDataset(
        num_images=num_images, seed=seed, image_size=image_size,
        num_shapes_range=(1, 1), shape_size_range=(15, 40),
        rotate_shapes=rotate, background=BackgroundType.SOLID,
        transform=transforms.ToTensor(),
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=rotate is False,
        collate_fn=ShapeDataset.collate_function, num_workers=0,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = (args.image_size, args.image_size)

    # Train WITHOUT rotation; validate WITH rotation. Two distributions, one knob apart.
    train_loader = make_loader(rotate=False, num_images=1000, seed=0,
                               image_size=image_size, batch_size=args.batch_size)
    val_loader = make_loader(rotate=True, num_images=500, seed=1234,
                             image_size=image_size, batch_size=args.batch_size)

    model = CenterHeatmapNet(num_classes=3, stride=4).to(device)
    loss_fn = HeatmapLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    w, h = image_size
    scale = torch.tensor([w, h], dtype=torch.float32, device=device)

    for epoch in range(args.epochs):
        model.train()
        for images, anns in train_loader:
            images = images.to(device)
            centers_px = torch.tensor(
                [a[0]["center"] for a in anns], dtype=torch.float32, device=device,
            ) * scale
            classes = torch.tensor(
                [a[0]["shape"] for a in anns], dtype=torch.long, device=device,
            )
            loss = loss_fn(model(images), centers_px, classes).total
            opt.zero_grad()
            loss.backward()
            opt.step()

    metrics = evaluate_heatmap(model, val_loader, device, image_size)
    print("clean-train -> rotated-val:")
    print(f"  median_center_px = {metrics['single/median_center_px']:.2f}")
    print(f"  accuracy         = {metrics['single/accuracy']:.3f}")
    print("Flip make_loader(rotate=True) for the train set to close the gap.")


if __name__ == "__main__":
    main()
