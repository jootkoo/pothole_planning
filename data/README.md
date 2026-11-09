# Pothole Mix binary segmentation data

`potmix.py` loads the main still-image dataset through `PotMixDataset`.
The separate RGB-D video and video-mask folders are not used by this loader.

## Verified local layout

Supply the root `potmix-dataset/pothole-mix-v1.0-20220526/pothole-mix`,
which contains `training`, `validation`, and `testing`. Each split contains
source-specific `images` and `masks` folders.

Filename pairing was checked for all **4,340** samples: **3,340 training,
496 validation, and 504 testing**. All image/mask headers were checked;
their stored dimensions match. The provided splits are preserved.
Pairing uses source folder and filename stem, including uppercase `.JPG`.
Sample IDs include split and source to avoid filename collisions.

Sources are `cnr-road-dataset`, `crack500`, `cracks-and-potholes-in-road`,
`edmcrack600`, `gaps384`, and `pothole600`.

**Pothole-600 is already part of Pothole Mix.** A later evaluation using it
must account for overlap; its full collection cannot be claimed as an
independent unseen dataset after training on Pothole Mix.

## Verified mask meaning

The supplied readme describes the dataset but does not specify mask colors.
The authors' `TargetMaskConvertTransform` in their
[evaluation notebook](https://gitlab.com/4ndr3aR/pothole-mix-segmentation/-/blob/main/shrec-22/evaluation/shrec-2022-evaluation.ipynb)
explicitly defines the mapping below. Representative local training masks
from all six sources were checked against it.

| RGB color | Dataset meaning | Binary target |
| --- | --- | ---: |
| `(255,0,0)` red | Pothole | 1 |
| `(0,255,0)` green | Crack | 0 |
| `(0,0,0)` black | Background/asphalt | 0 |

A nonzero-pixel threshold would incorrectly include cracks. The loader maps
exact colors and rejects unknown colors. Palette-mode masks are expanded to
RGB. Every mask is validated when loaded; the initial inspection did not
decode every mask's pixels.

## Baseline preprocessing choices

- Preserve stored pixel orientation for **both** files. The header audit
  found 320 images with nonidentity EXIF rotation tags. Rotating only an
  image would change its alignment with the corresponding mask.
- Convert images to RGB float32 `[0,1]`; do not use pretrained normalization.
- Resize RGB bilinearly and binary masks with nearest-neighbor interpolation
  to the same explicit `(height,width)`.
- Return image `(3,H,W)`, float32 binary mask `(1,H,W)`, `id`, and
  `original_size` (stored height/width).
- Apply no augmentation, cropping, letterboxing, or split reassignment.

Direct resizing can distort aspect ratio. It is a project baseline choice,
not a PotNet paper setting. Dimensions must each be >=64 to retain six pools.
The checker's 128×128 input is a development size, not a final training size.
Future mask/depth integration must restore the original coordinate system.

## Load a sample

Run Python from the project root:

```python
from data.potmix import DEFAULT_POTMIX_ROOT, PotMixDataset

training = PotMixDataset(DEFAULT_POTMIX_ROOT, "training", image_size=(128, 128))
sample = training[0]
print(len(training))           # 3340 for the complete local training split
print(sample["image"].shape)   # (3,128,128)
print(sample["mask"].shape)    # (1,128,128)
print(sample["id"])
```

Use validation/testing separately. A standard PyTorch `DataLoader` can batch
these dictionaries. The checker uses only training pixels for its model
check; validation/testing are enumerated for pair counts without fitting.

## Repeat the real-pair check

For optimizer-based training and desktop GPU setup, use the separate
[training guide](../training/README.md). On the desktop, install CUDA-enabled
PyTorch as described there before installing the remaining requirements.

From the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m scripts.check_segmentation --dropout-after-conv3 0.2 --dropout-after-conv5 0.3
```

The rates are explicit examples, not paper values. The command writes
`reports/potmix_training_preview.png`: RGB, original colored annotation, and
binary target, using one training example per source. White means pothole.
An all-black target is valid for a crack-only image. This is a ground-truth
preview, not model predictions.

Sources: the local `readme.txt`, the authors'
[dataset page](https://data.mendeley.com/datasets/kfth5g2xk3/2), and the linked
evaluation notebook. Original dataset files are read only.
