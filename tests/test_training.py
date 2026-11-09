"""Loss/metric math, real weight updates, checkpoint reload, and split isolation."""

import json
import math

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from models import PotNetSegmentation
from scripts.train_segmentation import main
from training.segmentation import (
    BinaryMetrics, BinarySegmentationLoss, evaluate, overfit_passed,
    save_checkpoint, select_overfit_items, train_batch,
)


@pytest.fixture(scope="module", autouse=True)
def few_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 2))
    yield
    torch.set_num_threads(previous)


def model_and_batch():
    torch.manual_seed(3)
    model = PotNetSegmentation(dropout_after_conv3=0.0, dropout_after_conv5=0.0)
    batch = {"image": torch.rand(2, 3, 64, 64),
             "mask": torch.randint(0, 2, (2, 1, 64, 64)).float(), "id": ["one", "two"]}
    return model, batch


def test_loss_matches_known_bce_and_soft_dice():
    logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
    target = torch.tensor([[[[1., 0.], [1., 0.]]]])
    loss = BinarySegmentationLoss()(logits, target)
    assert loss.item() == pytest.approx(math.log(2) + 0.5, abs=1e-6)
    assert BinarySegmentationLoss(0)(logits, target).item() == pytest.approx(math.log(2))
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_loss_stays_finite_for_extreme_wrong_logits():
    logits = torch.tensor([[[[1000., -1000.]]]], requires_grad=True)
    target = torch.tensor([[[[0., 1.]]]])
    loss = BinarySegmentationLoss()(logits, target)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_metrics_use_global_counts_instead_of_average_batch_iou():
    metrics = BinaryMetrics()
    metrics.update(torch.tensor([10., 10., -10.]), torch.tensor([1., 0., 0.]))
    metrics.update(torch.tensor([10.] * 9 + [-10.]), torch.ones(10))
    result = metrics.compute()
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (10, 1, 1, 1)
    assert result["iou"] == pytest.approx(10/12)
    assert result["dice"] == pytest.approx(20/22)


def test_background_only_masks_do_not_get_inflated_iou():
    metrics = BinaryMetrics()
    metrics.update(torch.full((4,), -10.), torch.zeros(4))
    result = metrics.compute()
    assert result["iou"] is None and result["dice"] is None
    assert result["tn"] == 4


def test_overfit_criterion_checks_each_positive_and_negative():
    initial = {"loss": 2.0}
    positive = {"tp": 95, "fp": 0, "fn": 5, "tn": 900, "iou": 0.95}
    negative = {"tp": 0, "fp": 0, "fn": 0, "tn": 1000, "iou": None}
    current = {"loss": 0.1, "iou": 0.95, "images": [positive, negative]}
    assert overfit_passed(initial, current)
    missed = {"tp": 0, "fp": 0, "fn": 1, "tn": 999, "iou": 0.0}
    assert not overfit_passed(initial, {**current, "images": [positive, negative, missed]})
    noisy_negative = {**negative, "fp": 2, "tn": 998}
    assert not overfit_passed(initial, {**current, "images": [positive, noisy_negative]})
    assert not overfit_passed(initial, {**current, "loss": 3.0})


def test_optimizer_updates_both_encoder_and_decoder():
    model, batch = model_and_batch()
    encoder_before = model.encoder.block1.conv.weight.detach().clone()
    decoder_before = model.decoder.head.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = train_batch(model, batch, optimizer, BinarySegmentationLoss(), "cpu")
    assert math.isfinite(loss)
    assert not torch.equal(encoder_before, model.encoder.block1.conv.weight)
    assert not torch.equal(decoder_before, model.decoder.head.weight)


def test_evaluation_does_not_update_weights_or_batchnorm():
    model, batch = model_and_batch()
    before = {key: tensor.clone() for key, tensor in model.state_dict().items()}
    result = evaluate(model, [batch], BinarySegmentationLoss(), "cpu", per_image=True)
    assert math.isfinite(result["loss"]) and len(result["images"]) == 2
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_checkpoint_reload_reproduces_predictions_and_optimizer(tmp_path):
    model, batch = model_and_batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_batch(model, batch, optimizer, BinarySegmentationLoss(), "cpu")
    config = {"dropout_after_conv3": 0.0, "dropout_after_conv5": 0.0}
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimizer, config, {"mode": "overfit"}, 1, {"iou": 0.2})
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    restored = PotNetSegmentation(**checkpoint["model_config"]).eval()
    restored.load_state_dict(checkpoint["model_state"])
    restored_optimizer = torch.optim.Adam(restored.parameters())
    restored_optimizer.load_state_dict(checkpoint["optimizer_state"])
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(batch["image"]), restored(batch["image"]))
    assert checkpoint["position"] == 1 and restored_optimizer.state_dict()["state"]
    assert not path.with_suffix(".pt.tmp").exists()


def test_overfit_selection_contains_positive_and_negative_targets():
    samples = [{"id": str(i), "mask": torch.full((1, 2, 2), value)}
               for i, value in enumerate((1., 1., 0., 0.))]
    selected = select_overfit_items(samples, 4)
    assert len(selected) == 4 and sum(bool(x["mask"].any()) for x in selected) == 2
    with pytest.raises(ValueError, match="enough positive and negative"):
        select_overfit_items(samples[:2], 2)


def test_train_cli_uses_separate_validation_and_never_testing(tmp_path, monkeypatch):
    import scripts.train_segmentation as script
    from data.potmix import PotMixDataset

    root = tmp_path / "data"
    for split in ("training", "validation"):
        for index in range(2):
            parent = root / split / "source"
            (parent / "images").mkdir(parents=True, exist_ok=True)
            (parent / "masks").mkdir(exist_ok=True)
            rgb = np.zeros((64, 64, 3), dtype=np.uint8)
            rgb[8:32, 8 + index:32 + index] = (255, 0, 0)
            Image.fromarray(rgb).save(parent / "images" / f"{index}.png")
            Image.fromarray(rgb).save(parent / "masks" / f"{index}.png")
    opened = []
    def tracked(root, split, image_size):
        opened.append(split)
        return PotMixDataset(root, split, image_size)
    monkeypatch.setattr(script, "PotMixDataset", tracked)
    output = tmp_path / "run"
    code = main(["--mode", "train", "--data-root", str(root), "--output", str(output),
                 "--dropout-after-conv3", "0", "--dropout-after-conv5", "0",
                 "--epochs", "1", "--train-limit", "1", "--val-limit", "1",
                 "--threads", "2"])
    assert code == 0 and opened == ["training", "validation"]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["test_split_used"] is False and summary["position"] == 1
    assert "validation" in summary["evaluation_split"]
    assert (output / "best.pt").is_file() and (output / "last.pt").is_file()


def test_explicit_cuda_request_fails_before_training_if_cuda_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "must_not_start"
    with pytest.raises(SystemExit) as error:
        main(["--device", "cuda", "--dropout-after-conv3", "0",
              "--dropout-after-conv5", "0", "--output", str(output)])
    assert error.value.code == 2 and not output.exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA-enabled PyTorch and NVIDIA GPU are required")
def test_cuda_training_and_evaluation_keep_tensors_on_gpu():
    model, batch = model_and_batch()
    model = model.to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = train_batch(model, batch, optimizer, BinarySegmentationLoss(), "cuda")
    result = evaluate(model, [batch], BinarySegmentationLoss(), "cuda")
    assert math.isfinite(loss) and math.isfinite(result["loss"])
    assert all(parameter.is_cuda and parameter.grad is not None and parameter.grad.is_cuda
               for parameter in model.parameters())
