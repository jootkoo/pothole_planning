"""Train PotNet segmentation, or first verify learning on a tiny training set.

overfit: deliberately memorize a few positive/negative TRAINING examples.
train: optimize the training split and select a checkpoint on validation IoU.
Neither mode reads the testing split. See training/README.md for commands.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import random
import time
import uuid

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader, Subset

from data.potmix import DEFAULT_POTMIX_ROOT, PotMixDataset
from models.potnet_decoder import DEFAULT_DECODER_CHANNELS, PotNetSegmentation
from training.segmentation import (
    BinarySegmentationLoss, evaluate, overfit_passed, save_checkpoint,
    select_overfit_items, train_batch, train_epoch,
)


def save_predictions(model, items, device, destination):
    """Show actual model predictions alongside inputs and ground-truth masks."""
    model.eval()
    with torch.inference_mode():
        # Render one image at a time so the preview does not secretly require
        # a larger GPU batch than the user's actual training batch size.
        probabilities = [torch.sigmoid(model(item["image"].unsqueeze(0).to(device)))[0, 0].cpu()
                         for item in items]
    tile, row_height = 192, 224
    canvas = Image.new("RGB", (tile * 4, 24 + row_height * len(items)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(("RGB input", "Ground truth", "Pothole probability", "Mask at >= 0.5")):
        draw.text((column * tile + 4, 4), title, fill="black")
    for row, item in enumerate(items):
        y = 24 + row * row_height
        draw.text((4, y), item["id"], fill="black")
        rgb = (item["image"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        target = (item["mask"][0].numpy() * 255).astype(np.uint8)
        prob = (probabilities[row].numpy() * 255).round().astype(np.uint8)
        predicted = ((probabilities[row].numpy() >= 0.5) * 255).astype(np.uint8)
        for column, array in enumerate((rgb, target, prob, predicted)):
            resample = Image.Resampling.BILINEAR if column in (0, 2) else Image.Resampling.NEAREST
            panel = Image.fromarray(array).convert("RGB").resize((tile, tile), resample)
            canvas.paste(panel, (column * tile, y + 24))
    canvas.save(destination)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("overfit", "train"), default="overfit")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_POTMIX_ROOT)
    parser.add_argument("--output", type=Path, help="New run directory; existing directories are not overwritten")
    parser.add_argument("--dropout-after-conv3", type=float, required=True)
    parser.add_argument("--dropout-after-conv5", type=float, required=True)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--decoder-channels", type=int, nargs=6, default=DEFAULT_DECODER_CHANNELS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200, help="Maximum optimizer updates in overfit mode")
    parser.add_argument("--overfit-samples", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=10, help="Overfit evaluation interval in updates")
    parser.add_argument("--target-iou", type=float, default=0.9, help="Diagnostic criterion, not a claimed benchmark")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs in train mode")
    parser.add_argument("--train-limit", type=int, help="Optional debug cap on training examples in train mode")
    parser.add_argument("--val-limit", type=int, help="Optional debug cap on validation examples")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("height", "width", "batch_size", "threads", "steps", "eval_every", "epochs"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive.")
    if min(args.height, args.width) < 64 or args.workers < 0 or args.overfit_samples < 2:
        parser.error("Use H/W >=64, workers >=0, and overfit-samples >=2.")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr must be positive and finite.")
    if not math.isfinite(args.target_iou) or not 0 < args.target_iou <= 1:
        parser.error("target-iou must be in (0,1].")
    if any(value is not None and value < 1 for value in (args.train_limit, args.val_limit)):
        parser.error("Debug dataset limits must be positive.")
    if args.mode == "overfit" and (args.train_limit or args.val_limit):
        parser.error("Use overfit-samples in overfit mode; limits are for train mode.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch installation; select cpu or auto.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    model_config = {
        "dropout_after_conv3": args.dropout_after_conv3,
        "dropout_after_conv5": args.dropout_after_conv5,
        "decoder_channels": tuple(args.decoder_channels),
    }
    # Constructor validation also checks the explicitly chosen dropout rates.
    model = PotNetSegmentation(**model_config).to(device)
    criterion = BinarySegmentationLoss(args.dice_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    training = PotMixDataset(args.data_root, "training", (args.height, args.width))

    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {"batch_size": args.batch_size, "num_workers": args.workers,
                      "pin_memory": device.type == "cuda"}
    if args.mode == "overfit":
        items = select_overfit_items(training, args.overfit_samples)
        train_loader = DataLoader(items, shuffle=True, generator=generator, **loader_options)
        eval_loader = DataLoader(items, shuffle=False, **loader_options)
        preview_items = items
        evaluation_split = "same selected training examples (memorization diagnostic)"
    else:
        validation = PotMixDataset(args.data_root, "validation", (args.height, args.width))
        train_data = Subset(training, range(min(args.train_limit, len(training)))) if args.train_limit else training
        val_data = Subset(validation, range(min(args.val_limit, len(validation)))) if args.val_limit else validation
        train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_options)
        eval_loader = DataLoader(val_data, shuffle=False, **loader_options)
        preview_items = [val_data[i] for i in range(min(4, len(val_data)))]
        evaluation_split = "official validation split" + (" (DEBUG SUBSET)" if args.val_limit else "")

    output = args.output or Path("runs") / f"{args.mode}_{uuid.uuid4().hex[:8]}"
    if output.exists():
        parser.error(f"Output directory already exists: {output}. Choose a new --output.")
    output.mkdir(parents=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({
        "output": str(output.resolve()), "data_root": str(args.data_root.resolve()),
        "device": str(device), "torch_version": str(torch.__version__),
        "cuda_build": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "training_examples": len(train_loader.dataset), "evaluation_examples": len(eval_loader.dataset),
        "evaluation_split": evaluation_split, "threshold": 0.5,
        "selected_ids": [item["id"] for item in items] if args.mode == "overfit" else None,
    })
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Mode: {args.mode}; device: {device}; output: {output.resolve()}", flush=True)
    print(f"Train examples: {len(train_loader.dataset)}; evaluate: {evaluation_split}", flush=True)
    if args.mode == "overfit":
        for item in items:
            print(f"  {item['id']}: {int(item['mask'].sum())} pothole pixels", flush=True)

    per_image = args.mode == "overfit"
    initial = evaluate(model, eval_loader, criterion, device, per_image=per_image)
    save_predictions(model, preview_items, device, output / "before.png")
    best_rank = (-1, -1.0)
    best_metrics, best_position = None, None
    passed = False
    started = time.perf_counter()
    csv_fields = ("position", "train_loss", "eval_loss", "iou", "dice", "precision", "recall")

    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()

        def record(position, train_loss, result):
            nonlocal best_rank, best_metrics, best_position, passed
            writer.writerow({"position": position, "train_loss": train_loss, "eval_loss": result["loss"],
                             **{name: result[name] for name in ("iou", "dice", "precision", "recall")}})
            stream.flush()
            passed = per_image and overfit_passed(initial, result, args.target_iou)
            rank = (int(passed), result["iou"] if result["iou"] is not None else -1.0)
            if position > 0 and rank > best_rank:
                best_rank, best_metrics, best_position = rank, result, position
                # best.pt is a light model checkpoint; last.pt also keeps Adam state.
                save_checkpoint(output / "best.pt", model, None, model_config, config, position, result)
            iou_text = f"{result['iou']:.4f}" if result["iou"] is not None else "undefined"
            unit = "update" if per_image else "epoch"
            print(f"{unit} {position}: eval_loss={result['loss']:.6f}, pothole_IoU={iou_text}", flush=True)

        record(0, None, initial)
        if per_image:
            iterator = iter(train_loader)
            for position in range(1, args.steps + 1):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    batch = next(iterator)
                train_loss = train_batch(model, batch, optimizer, criterion, device)
                if position % args.eval_every == 0 or position == args.steps:
                    current = evaluate(model, eval_loader, criterion, device, per_image=True)
                    record(position, train_loss, current)
                    if passed:
                        break
        else:
            for position in range(1, args.epochs + 1):
                def show_progress(batch_number, average_loss):
                    if batch_number % 25 == 0 or batch_number == len(train_loader):
                        print(f"epoch {position}, batch {batch_number}/{len(train_loader)}: "
                              f"training_loss={average_loss:.6f}", flush=True)
                train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                                         on_batch=show_progress)
                current = evaluate(model, eval_loader, criterion, device)
                record(position, train_loss, current)
                save_checkpoint(output / "last.pt", model, optimizer, model_config, config, position, current)

    if per_image:
        save_checkpoint(output / "last.pt", model, optimizer, model_config, config, position, current)
    save_predictions(model, preview_items, device, output / "after.png")
    summary = {
        "mode": args.mode, "position": position, "position_unit": "optimizer update" if per_image else "epoch",
        "evaluation_split": evaluation_split, "initial": initial, "final": current,
        "best": best_metrics, "best_position": best_position,
        "overfit_passed": passed if per_image else None,
        "target_iou": args.target_iou if per_image else None,
        "elapsed_seconds": time.perf_counter() - started,
        "full_validation_used": args.mode == "train" and not args.val_limit,
        "test_split_used": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if per_image:
        print("PASS: tiny-set memorization criteria met." if passed else
              "NOT YET PASSED: tiny-set memorization criteria not reached within the update budget.", flush=True)
        print("These are training-subset results, not held-out accuracy.", flush=True)
    print(f"Saved config, history, checkpoints, before/after previews and summary in {output.resolve()}", flush=True)
    return 0 if args.mode == "train" or passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
