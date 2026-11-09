"""Verify the binary loader and full decoder on a real training pair.

No optimizer step, checkpoint, training epoch or accuracy evaluation is run.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn

from data.potmix import DEFAULT_POTMIX_ROOT, PotMixDataset
from models.potnet_decoder import PotNetSegmentation


def save_preview(dataset: PotMixDataset, destination: Path):
    """One training example per source: RGB, original labels, binary target."""
    examples = {}
    for index, pair in enumerate(dataset.samples):
        source = pair.image.parent.parent.name
        if source not in examples:
            examples[source] = index
    tile = 192
    canvas = Image.new("RGB", (tile * 3, (tile + 36) * len(examples) + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(("RGB input", "Dataset labels", "Binary pothole target")):
        draw.text((column * tile + 4, 4), title, fill="black")
    for row, (source, index) in enumerate(examples.items()):
        item = dataset[index]
        rgb = Image.fromarray((item["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        with Image.open(dataset.samples[index].mask) as mask:
            labels = mask.convert("RGB").resize((tile, tile), Image.Resampling.NEAREST)
        binary = Image.fromarray((item["mask"][0].numpy() * 255).astype(np.uint8)).convert("RGB")
        top = 24 + row * (tile + 36)
        draw.text((4, top), source, fill="black")
        canvas.paste(rgb.resize((tile, tile), Image.Resampling.BILINEAR), (0, top + 24))
        canvas.paste(labels, (tile, top + 24))
        canvas.paste(binary.resize((tile, tile), Image.Resampling.NEAREST), (tile * 2, top + 24))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_POTMIX_ROOT)
    parser.add_argument("--dropout-after-conv3", type=float, required=True)
    parser.add_argument("--dropout-after-conv5", type=float, required=True)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview", type=Path, default=Path("reports/potmix_training_preview.png"))
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        parser.error("threads and batch-size must be positive.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    datasets = {
        split: PotMixDataset(args.data_root, split, (args.height, args.width))
        for split in ("training", "validation", "testing")
    }
    for split, dataset in datasets.items():
        print(f"{split}: {len(dataset):,} filename-paired samples")
    dataset = datasets["training"]
    if args.batch_size > len(dataset):
        parser.error("batch-size exceeds the training split length.")
    save_preview(dataset, args.preview)
    items = [dataset[i] for i in range(args.batch_size)]
    images = torch.stack([item["image"] for item in items])
    targets = torch.stack([item["mask"] for item in items])
    model = PotNetSegmentation(
        dropout_after_conv3=args.dropout_after_conv3,
        dropout_after_conv5=args.dropout_after_conv5,
    ).train()
    logits = model(images)
    assert logits.shape == targets.shape
    assert torch.isfinite(logits).all()
    loss = nn.BCEWithLogitsLoss()(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    print(f"Training sample: {items[0]['id']}")
    print(f"Images: {tuple(images.shape)}; targets: {tuple(targets.shape)}")
    print(f"Target values: {torch.unique(targets).tolist()} (red potholes=1, others=0)")
    print(f"Raw logits: {tuple(logits.shape)}")
    print(f"Sanity-check BCE loss: {loss.item():.6f} (not an accuracy measurement)")
    print(f"Preview: {args.preview.resolve()}")
    print("PASS: real-pair loading, output shape, finite loss, and encoder/decoder gradients")
    print("No optimizer update or checkpoint was produced; the model is still untrained.")


if __name__ == "__main__":
    main()
