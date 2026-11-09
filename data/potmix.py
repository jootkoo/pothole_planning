"""Binary pothole targets from the verified Pothole Mix RGB mask palette."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


DEFAULT_POTMIX_ROOT = Path(
    "potmix-dataset/pothole-mix-v1.0-20220526/pothole-mix"
)
MASK_COLORS = {(0, 0, 0): "background", (255, 0, 0): "pothole", (0, 255, 0): "crack"}


@dataclass(frozen=True)
class ImageMaskPair:
    image: Path
    mask: Path
    sample_id: str


def binary_pothole_mask(mask: Image.Image) -> np.ndarray:
    """Red -> 1; green/black -> 0. Reject unknown colors instead of guessing.

    Palette-mode masks are expanded to their RGB colors before interpretation.
    Mapping source: authors' TargetMaskConvertTransform in their evaluation
    notebook; see data/README.md. This function does not threshold luminance.
    """
    rgb = np.asarray(mask.convert("RGB"))
    red = (rgb[..., 0] == 255) & (rgb[..., 1] == 0) & (rgb[..., 2] == 0)
    green = (rgb[..., 0] == 0) & (rgb[..., 1] == 255) & (rgb[..., 2] == 0)
    black = (rgb == 0).all(axis=-1)
    unknown = ~(red | green | black)
    if unknown.any():
        example = tuple(int(value) for value in rgb[unknown][0])
        raise ValueError(f"Unknown mask color {example}; expected red, green, or black.")
    return red.astype(np.uint8)


class PotMixDataset(Dataset):
    """Read one existing split without combining or repartitioning its samples.

    root must contain training/, validation/, testing/, each with source-specific
    images/ and masks/ directories. image_size is (height, width), each >= 64.
    RGB is scaled to float32 [0,1]. RGB uses bilinear resize; binary targets
    use nearest-neighbor. Direct resizing can distort aspect ratio: it is an
    explicit baseline choice, not a reported PotNet preprocessing recipe.

    Both files are read in stored pixel orientation. In particular, do not
    EXIF-transpose only the image: that would misalign many Crack500 masks.
    """

    def __init__(self, root: str | Path, split: str, image_size: tuple[int, int]):
        self.root = Path(root)
        if split not in {"training", "validation", "testing"}:
            raise ValueError("split must be training, validation, or testing.")
        if len(image_size) != 2 or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 64 for v in image_size
        ):
            raise ValueError("image_size must contain two integers >= 64 (height, width).")
        self.image_size = tuple(image_size)
        split_root = self.root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_root}")
        pairs = []
        for image_dir in sorted(split_root.glob("*/images")):
            mask_dir = image_dir.parent / "masks"
            images = {}
            for image in sorted(image_dir.iterdir()):
                if image.is_file() and image.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    if image.stem in images:
                        raise ValueError(f"Duplicate image stem in {image_dir}: {image.stem}")
                    images[image.stem] = image
            masks = {p.stem: p for p in mask_dir.glob("*.png")}
            if images.keys() != masks.keys():
                raise ValueError(
                    f"Unpaired files in {image_dir.parent}: "
                    f"missing masks={sorted(images.keys() - masks.keys())[:5]}, "
                    f"orphan masks={sorted(masks.keys() - images.keys())[:5]}"
                )
            for stem, image in images.items():
                pairs.append(ImageMaskPair(
                    image, masks[stem], f"{split}/{image_dir.parent.name}/{stem}"
                ))
        if not pairs:
            raise ValueError(f"No image/mask pairs found in {split_root}")
        self.samples = tuple(pairs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pair = self.samples[index]
        with Image.open(pair.image) as source, Image.open(pair.mask) as target:
            if source.size != target.size:
                raise ValueError(f"Image/mask dimensions differ for {pair.sample_id}")
            original_size = (source.height, source.width)
            # Intentionally preserve stored orientation for both image and mask.
            rgb = source.convert("RGB")
            binary = Image.fromarray(binary_pothole_mask(target))
            size = (self.image_size[1], self.image_size[0])  # PIL expects (W,H).
            rgb = rgb.resize(size, resample=Image.Resampling.BILINEAR)
            binary = binary.resize(size, resample=Image.Resampling.NEAREST)
            image_array = np.array(rgb, dtype=np.float32) / 255.0
            mask_array = np.array(binary, dtype=np.float32)
        return {
            "image": torch.from_numpy(image_array).permute(2, 0, 1).contiguous(),
            "mask": torch.from_numpy(mask_array).unsqueeze(0),
            "id": pair.sample_id,
            "original_size": torch.tensor(original_size),
        }
