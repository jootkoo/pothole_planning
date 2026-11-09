"""Small, reusable training components for the binary segmentation baseline."""

import math
from pathlib import Path

import torch
from torch import nn


class BinarySegmentationLoss(nn.Module):
    """BCE on raw logits plus weighted soft Dice over all pixels in a batch.

    The loss choice/weight are project settings, not reported PotNet settings.
    BCE is numerically stable on logits; only the Dice branch uses sigmoid.
    """
    def __init__(self, dice_weight: float = 1.0):
        super().__init__()
        if not math.isfinite(dice_weight) or dice_weight < 0:
            raise ValueError("dice_weight must be finite and nonnegative.")
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        if logits.shape != targets.shape or logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError("Logits and targets must have matching (N,1,H,W) shapes.")
        bce = self.bce(logits, targets)
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * targets).sum()
        soft_dice = (2 * intersection + 1e-6) / (probabilities.sum() + targets.sum() + 1e-6)
        return bce + self.dice_weight * (1 - soft_dice)


class BinaryMetrics:
    """Global pixel counts, not an average of per-batch IoUs.

    None means an undefined zero denominator, e.g. no foreground union.
    This avoids giving empty background-only masks an inflated IoU of one.
    """
    def __init__(self, threshold=0.5):
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1.")
        self.threshold = threshold
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, logits, targets):
        if logits.shape != targets.shape:
            raise ValueError("Logit/target metric shapes must match.")
        predictions = torch.sigmoid(logits.detach()) >= self.threshold
        truth = targets.bool()
        self.tp += int((predictions & truth).sum().item())
        self.fp += int((predictions & ~truth).sum().item())
        self.fn += int((~predictions & truth).sum().item())
        self.tn += int((~predictions & ~truth).sum().item())

    def compute(self):
        def ratio(numerator, denominator):
            return numerator / denominator if denominator else None
        return {
            "iou": ratio(self.tp, self.tp + self.fp + self.fn),
            "dice": ratio(2 * self.tp, 2 * self.tp + self.fp + self.fn),
            "precision": ratio(self.tp, self.tp + self.fp),
            "recall": ratio(self.tp, self.tp + self.fn),
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
        }


def train_batch(model, batch, optimizer, criterion, device):
    """One real weight update: clear gradients, forward, backward, step."""
    model.train()  # Enable training behavior in encoder BN and dropout.
    # CUDA loaders pin CPU memory, allowing transfers to run asynchronously.
    images = batch["image"].to(device, non_blocking=True)
    targets = batch["mask"].to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = criterion(logits, targets)
    if not torch.isfinite(loss):
        raise FloatingPointError("Training loss became non-finite.")
    loss.backward()
    optimizer.step()
    return loss.item()


def train_epoch(model, loader, optimizer, criterion, device, *, on_batch=None):
    """Update on the training split; report loss weighted by sample count."""
    loss_sum, count = 0.0, 0
    for batch_number, batch in enumerate(loader, start=1):
        loss_sum += train_batch(model, batch, optimizer, criterion, device) * len(batch["image"])
        count += len(batch["image"])
        if on_batch is not None:
            on_batch(batch_number, loss_sum / count)
    if not count:
        raise ValueError("Training loader is empty.")
    return loss_sum / count


def evaluate(model, loader, criterion, device, *, per_image=False):
    """Evaluate without parameter/BN updates. Caller chooses which split.

    Overfit diagnostics deliberately evaluate the same few training images.
    Normal training passes the separate official validation split instead.
    """
    model.eval()
    metrics = BinaryMetrics()
    loss_sum, count, records = 0.0, 0, []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Evaluation loss became non-finite.")
            metrics.update(logits, targets)
            loss_sum += loss.item() * len(images)
            count += len(images)
            if per_image:
                for index in range(len(images)):
                    single = BinaryMetrics()
                    single.update(logits[index:index+1], targets[index:index+1])
                    records.append({"id": batch["id"][index], **single.compute()})
    if not count:
        raise ValueError("Evaluation loader is empty.")
    result = {"loss": loss_sum / count, **metrics.compute()}
    if per_image:
        result["images"] = records
    return result


def overfit_passed(initial, current, target_iou=0.9):
    """Require actual pothole overlap, a loss drop, and few negative false positives."""
    if current["iou"] is None or current["iou"] < target_iou or current["loss"] >= initial["loss"]:
        return False
    saw_positive = saw_negative = False
    for image in current["images"]:
        positives = image["tp"] + image["fn"]
        if positives:
            saw_positive = True
            if image["iou"] is None or image["iou"] < target_iou:
                return False
        else:
            saw_negative = True
            # Allow <=0.1% foreground errors on a background-only sample.
            if image["fp"] / (image["fp"] + image["tn"]) > 0.001:
                return False
    return saw_positive and saw_negative


def select_overfit_items(dataset, count):
    """Cache a fixed balanced set from training, selected using ground truth.

    A diagnostic set, not a representative evaluation set. No validation/test
    samples are used. Sampling order and selected IDs are recorded by the CLI.
    """
    if count < 2:
        raise ValueError("Overfit count must be at least two (positive and negative).")
    positive, negative = [], []
    needed_positive, needed_negative = (count + 1) // 2, count // 2
    for index in range(len(dataset)):
        item = dataset[index]
        if item["mask"].any():
            if len(positive) < needed_positive:
                positive.append(item)
        elif len(negative) < needed_negative:
            negative.append(item)
        if len(positive) == needed_positive and len(negative) == needed_negative:
            return positive + negative
    raise ValueError("Could not find enough positive and negative training targets for the diagnostic.")


def save_checkpoint(path, model, optimizer, model_config, run_config, position, metrics):
    """Save tensors and simple metadata, not a pickled model class instance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "model_config": model_config,
        "run_config": run_config,
        "position": position,
        "metrics": metrics,
        "torch_version": str(torch.__version__),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
