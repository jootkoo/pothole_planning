# PotNet encoder and project segmentation decoder

`potnet_encoder.py` is the existing paper-based encoder.
`potnet_decoder.py` adds `PotNetDecoder` and the complete
`PotNetSegmentation` model. All weights start from scratch.

The decoder source begins with the requested triple-single-quoted explanation
and now has beginner comments throughout. The [training guide](../training/README.md)
covers real weight updates, checkpoints, the completed four-image diagnostic,
and desktop CUDA setup.

## New decoder design

The decoder is **our baseline design**, not part of the PotNet paper.
The encoder's seven convolutions, six pools, operation order, and feature
interface are preserved.

Each decoder stage bilinearly upsamples to the actual skip map dimensions
(`align_corners=False`), concatenates channels, and applies two bias-enabled
3×3 convolutions with padding 1, each followed by ReLU.

| Stage | Encoder skip | Nominal stride | Default decoder channels |
| --- | --- | ---: | ---: |
| 1 | `conv7` | 32 | 128 |
| 2 | `conv5` | 16 | 96 |
| 3 | `conv4` | 8 | 64 |
| 4 | `conv3` | 4 | 48 |
| 5 | `conv2` | 2 | 32 |
| 6 | `conv1` | 1 | 16 |

The starting tensor is the final encoder output at stride64. Conv6 already
feeds conv7 and shares its size; it is not an extra upsampling scale.
A final 1×1 convolution returns one channel of **raw logits**, `(N,1,H,W)`.
Actual skip dimensions ensure alignment for odd or rectangular inputs.

Widths are configurable with a six-element `decoder_channels` tuple.
No BN, dropout, attention, or residual blocks are added in the decoder.
PyTorch defaults initialize all modules. These are project choices requiring
evaluation, not paper-reported settings or demonstrated improvements.

## Use the complete model

```python
import torch
from models import PotNetSegmentation

# Explicit experiment choices, not dropout rates reported in the paper.
model = PotNetSegmentation(dropout_after_conv3=0.2, dropout_after_conv5=0.3)
images = torch.randn(1, 3, 128, 128)
targets = torch.zeros(1, 1, 128, 128)
logits = model(images)
loss = torch.nn.BCEWithLogitsLoss()(logits, targets)
loss.backward()  # Gradients reach the encoder and decoder.
```

For later inference, use `model.eval()`, disable gradient tracking, and apply
sigmoid to the logits to obtain probabilities, then a chosen threshold for a
mask. Do not apply sigmoid before `BCEWithLogitsLoss`. Predictions need a
trained model to be useful.

## Verify the baseline

From the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.check_segmentation --dropout-after-conv3 0.2 --dropout-after-conv5 0.3
```

The real-pair checker defaults to 128×128, batch1, two CPU threads and seed0.
Override `--data-root`, `--height`, `--width`, `--batch-size`, `--threads`,
`--seed`, or `--preview` as needed; both dropout arguments are required.
It prints split counts, shapes, and a BCE sanity-check loss and verifies
finite gradients for every parameter. It performs no optimizer update,
checkpoint saving, training epoch, or accuracy evaluation.

The first run passed **50 tests**. A real training pair produced
`(1,1,128,128)` logits with finite loss and gradients. This verifies the
implementation, not segmentation quality.

See [data/README.md](../data/README.md) for labeling/loading and
[PAPER_EXACT_VS_ASSUMED.md](../PAPER_EXACT_VS_ASSUMED.md) for encoder provenance.
Training, checkpoints, pixel metrics, and the small-subset overfit check are
now implemented. The four-image diagnostic reached IoU0.918848 at update160;
this is memorization, not held-out performance. See the training guide for
commands and artifacts. Full training is planned for the desktop after its
graphics card and CUDA availability have been checked.
