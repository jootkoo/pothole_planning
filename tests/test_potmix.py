"""Synthetic fixtures keep loader tests independent of the downloaded dataset."""

import numpy as np
from PIL import Image
import pytest
import torch

from data.potmix import PotMixDataset, binary_pothole_mask


def make_pair(root, split="training", source="source", name="001", suffix=".png", colors=None):
    parent = root / split / source
    (parent / "images").mkdir(parents=True, exist_ok=True)
    (parent / "masks").mkdir(exist_ok=True)
    if colors is None:
        colors = np.zeros((64, 80, 3), dtype=np.uint8)
        colors[:, :20] = (255, 0, 0)
        colors[:, 20:40] = (0, 255, 0)
    image = parent / "images" / (name + suffix)
    mask = parent / "masks" / (name + ".png")
    Image.fromarray(colors).save(image)
    Image.fromarray(colors).save(mask)
    return image, mask


def test_color_mapping_is_potholes_only():
    rgb = np.array([[(255, 0, 0), (0, 255, 0), (0, 0, 0)]], dtype=np.uint8)
    assert binary_pothole_mask(Image.fromarray(rgb)).tolist() == [[1, 0, 0]]
    with pytest.raises(ValueError, match="Unknown mask color"):
        binary_pothole_mask(Image.new("RGB", (2, 2), "blue"))


def test_loader_resize_binary_values_and_rgb_scaling(tmp_path):
    make_pair(tmp_path)
    dataset = PotMixDataset(tmp_path, "training", (65, 96))
    item = dataset[0]
    assert item["image"].shape == (3, 65, 96)
    assert item["mask"].shape == (1, 65, 96)
    assert item["image"].dtype == item["mask"].dtype == torch.float32
    assert 0 <= item["image"].min() <= item["image"].max() <= 1
    assert set(torch.unique(item["mask"]).tolist()) == {0.0, 1.0}
    assert torch.all(item["mask"][0, :, :24] == 1)
    assert torch.all(item["mask"][0, :, 24:] == 0)
    assert item["original_size"].tolist() == [64, 80]


def test_source_names_and_splits_prevent_filename_collisions(tmp_path):
    make_pair(tmp_path, source="first")
    make_pair(tmp_path, source="second", suffix=".JPG")
    make_pair(tmp_path, split="validation", source="first", name="002")
    training = PotMixDataset(tmp_path, "training", (64, 64))
    validation = PotMixDataset(tmp_path, "validation", (64, 64))
    assert len(training) == 2 and len(validation) == 1
    assert len({pair.sample_id for pair in training.samples + validation.samples}) == 3
    assert all(pair.sample_id.startswith("training/") for pair in training.samples)


def test_unpaired_and_mismatched_files_fail(tmp_path):
    _, mask = make_pair(tmp_path)
    mask.unlink()
    with pytest.raises(ValueError, match="Unpaired"):
        PotMixDataset(tmp_path, "training", (64, 64))
    Image.new("RGB", (30, 40)).save(mask)
    with pytest.raises(ValueError, match="dimensions differ"):
        PotMixDataset(tmp_path, "training", (64, 64))[0]


def test_exif_orientation_does_not_rotate_image_without_mask(tmp_path):
    image, _ = make_pair(tmp_path)
    with Image.open(image) as source:
        exif = Image.Exif()
        exif[274] = 6
        source.save(image, exif=exif)
    item = PotMixDataset(tmp_path, "training", (64, 80))[0]
    assert torch.equal(item["image"][0] == 1, item["mask"][0] == 1)


@pytest.mark.parametrize("size", [(63, 64), (64, 0), (64, 64.5)])
def test_invalid_resize_size(tmp_path, size):
    with pytest.raises(ValueError, match="image_size"):
        PotMixDataset(tmp_path, "training", size)


def test_invalid_split_and_missing_dataset(tmp_path):
    with pytest.raises(ValueError, match="split"):
        PotMixDataset(tmp_path, "train", (64, 64))
    with pytest.raises(FileNotFoundError):
        PotMixDataset(tmp_path, "training", (64, 64))
