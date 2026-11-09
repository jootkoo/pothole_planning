# Training the PotNet segmentation model

This milestone adds real optimizer updates, validation, pixel metrics, saved
weights, and a small-subset learning diagnostic. The decoder has a beginner
explanation in its opening triple-quoted text and comments throughout
[models/potnet_decoder.py](../models/potnet_decoder.py).

## What the files do

| File | Purpose |
| --- | --- |
| `scripts/train_segmentation.py` | Command-line entry point for overfit and normal training modes |
| `training/segmentation.py` | Loss, pixel metrics, optimization, evaluation, sample selection, and checkpoint helpers |
| `tests/test_training.py` | Math checks, weight updates, checkpoint reload, validation isolation, and a conditional CUDA test |

Both model parts are optimized together. No pretrained weights are loaded.
The encoder architecture and decoder widths are preserved. The trainer uses
the verified binary [Pothole Mix loader](../data/README.md).

## 1. Verify learning on a handful of images

From the project root in PowerShell, using the existing CPU environment:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_segmentation --mode overfit --device cpu --dropout-after-conv3 0 --dropout-after-conv5 0 --height 64 --width 64 --batch-size 4 --overfit-samples 4 --steps 200 --output runs/overfit_check
```

The output directory must be new. Pick a different name for another run, or
omit `--output` to generate a unique name automatically.

Overfit mode caches a fixed set with both positive pothole masks and negative
masks from **training only**. For four examples, it selects two of each in
dataset order. This deliberate ground-truth selection is a wiring diagnostic,
not a representative evaluation sample. It repeatedly updates model weights
on that set and evaluates the same set in `eval()` mode every ten updates.
Dropout is explicitly disabled in this diagnostic to make memorization easier;
those zeros are not reported paper settings or prescribed full-training rates.

The diagnostic stops successfully only when:

- Global pothole IoU is at least `--target-iou` (default 0.90).
- Every positive image independently reaches that IoU target.
- Each negative image has at most 0.1% false-positive pixels.
- Evaluation loss is below its initial value.

It returns exit code 2 if the update budget expires without meeting these
criteria, while still saving the result. This is reported as not yet passed,
not as a successful accuracy result. `--steps` counts optimizer updates.
`--eval-every`, `--overfit-samples`, and `--target-iou` configure the check.
The 64×64 size is a cheap development choice, not the final training size.

## Completed CPU diagnostic

The first real run is saved in `runs/overfit_4x64_v1/`, with seed0, four
64×64 training images, batch4, dropout0/0, Adam LR0.001, and Dice weight1.

| Quantity | Before updates | At update 160 |
| --- | ---: | ---: |
| BCE + soft Dice loss | 1.675878 | 0.055940 |
| Pothole pixel IoU | 0.022827 | 0.918848 |
| Pothole pixel Dice | 0.044635 | 0.957708 |
| Precision | 0.022827 | 0.977716 |
| Recall | 1.000000 | 0.938503 |

The two positive-image IoUs were 0.907975 and 0.982143. Both negative images
had zero false-positive pixels. The diagnostic passed after 160 updates.
Reloading `last.pt` reproduced the recorded metrics. Inspect `before.png`
and `after.png` for RGB, ground truth, probability, and thresholded predictions.

**These numbers describe memorization of those four training images. They
do not measure performance on unseen images.** The sanity-check weights are
not a generally trained pothole model. A fresh normal training command starts
a new model rather than automatically loading those diagnostic weights.

## 2. Set up the desktop for GPU training

The desktop's graphics card and operating system still need to be checked.
An AMD Ryzen5 is the CPU; 32–64GB is system RAM. CUDA training needs a supported
NVIDIA graphics card and its separate video memory (VRAM). A Ryzen CPU can be
paired with an NVIDIA GPU. Choose the CUDA build after identifying the card
and driver, using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

Copy the source folders (`models`, `data`, `training`, `scripts`, `tests`,
`configs`), requirements and documentation. Copy the dataset separately or
point `--data-root` at its new location. Create a **new environment on the
desktop**; do not transfer the current CPU-only `.venv`.

Windows PowerShell setup:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Next, use the official selector for the desktop OS, Pip, Python and the
appropriate CUDA platform. Run its install command using this environment's
Python: replace the command's `pip`/`pip3` prefix with
`.\.venv\Scripts\python.exe -m pip`. This step installs the CUDA-enabled
PyTorch build. Do not use the CPU-wheel install command from the local CPU
setup when preparing this GPU environment.

Then install remaining project dependencies and verify the GPU:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU detected')"
.\.venv\Scripts\python.exe -m pytest -q
```

On Linux, create the environment with `python3 -m venv .venv` and use
`./.venv/bin/python` instead of the Windows executable in these commands.
Select Linux in the official installer. Exact CUDA installation commands
are intentionally deferred until the desktop's card and driver are known.

Use `--device cuda` to require the GPU: the program errors before training
if CUDA is unavailable. `--device auto` (the default) chooses CUDA when
available and otherwise reports that it selected CPU. `--device cpu` forces
CPU. Model weights, inputs and targets are moved to the selected device.
CUDA loaders pin host memory for asynchronous transfers. Run metadata records
the selected device, PyTorch/CUDA build and detected GPU name.

Repeat the diagnostic on the desktop first:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_segmentation --mode overfit --device cuda --dropout-after-conv3 0 --dropout-after-conv5 0 --height 64 --width 64 --batch-size 4 --overfit-samples 4 --steps 400 --output runs/desktop_overfit
```

Seeds and settings are recorded, but different hardware/builds may learn at
different rates. Start with a small batch/resolution and choose larger values
after seeing the desktop's VRAM use and validation results. The current loop
uses float32 training; mixed precision, multi-GPU training and distributed
training are not implemented in this baseline.

## 3. Train against the official validation split

Once desktop CUDA is working, this is an example full-training command:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_segmentation --mode train --device cuda --dropout-after-conv3 0.2 --dropout-after-conv5 0.3 --height 128 --width 128 --batch-size 4 --epochs 20 --output runs/potnet_baseline
```

By default it optimizes all 3,340 training examples, validates on the separate
496-example validation split after each epoch, and selects `best.pt` by
validation pothole IoU. The 504-example testing split is never opened by this
trainer. Keep it for a later final evaluation. The example resolution, epochs,
dropout, loss and LR are initial experiment settings, not demonstrated optima
or a segmentation recipe from the PotNet paper.

`--data-root` overrides the dataset path. `--workers` defaults to0 for a simple
Windows-compatible setup; increase it if data loading becomes the bottleneck.
`--threads` controls CPU work, not GPU compute. Each training epoch includes
forward passes, gradients, and Adam updates; validation does none of those
updates and uses the model's evaluation behavior.

For a short training/validation wiring check instead of a full run:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_segmentation --mode train --device cpu --dropout-after-conv3 0.2 --dropout-after-conv5 0.3 --height 64 --width 64 --batch-size 2 --epochs 1 --train-limit 4 --val-limit 2 --output runs/train_val_check
```

That limited real-data check was executed successfully. Its two validation
examples and one epoch are insufficient for a quality conclusion. Omit both
debug limits for a normal experiment. The loop prints one validation summary
per epoch; a full CPU epoch will take much longer than this diagnostic.

## Loss and metrics

The project baseline loss is `BCEWithLogitsLoss + dice_weight * soft_Dice_loss`
with default `dice_weight=1`. BCE takes raw logits. Sigmoid is used separately
inside the soft Dice calculation. This batch-level Dice uses all pixels in
the batch, with epsilon1e-6. Adam defaults to LR0.001 and zero weight decay.
Choose `--lr` and `--dice-weight` explicitly for comparisons.

Hard masks use probability >=0.5. IoU, Dice, precision and recall are computed
from accumulated global pixel counts, not the mean of batch IoUs. A zero
denominator is reported as JSON `null`/CSV blank, e.g. correct empty masks
have no foreground IoU. This avoids letting many empty masks inflate a score.
Overfit reports include per-image counts and a separate negative-image check.

## Run outputs and checkpoints

| Output | Meaning |
| --- | --- |
| `config.json` | Arguments, device/build, dataset sizes, and selected diagnostic IDs |
| `history.csv` | Initial evaluation and subsequent train/evaluation losses and metrics |
| `best.pt` | Model state plus configuration/metrics, selected by validation IoU in train mode |
| `last.pt` | Latest model state, Adam state and metadata; refreshed after every normal training epoch |
| `before.png`, `after.png` | Inputs, true masks, probabilities and thresholded predictions |
| `summary.json` | Initial/final/best measurements, pass status, and elapsed time |

In overfit mode the first checkpoint satisfying all diagnostic criteria takes
priority over a non-passing checkpoint; otherwise `best.pt` tracks subset IoU.
Checkpoints use `state_dict` data rather than serializing a whole model object.
The best checkpoint omits optimizer state; the last checkpoint includes it.
There is no command-line resume option yet. Adam state is stored for explicit
continuation code; this baseline CLI starts fresh for every new run.

Load a saved model on any supported device:

```python
import torch
from models import PotNetSegmentation

checkpoint = torch.load("runs/overfit_4x64_v1/last.pt", map_location="cpu", weights_only=True)
model = PotNetSegmentation(**checkpoint["model_config"])
model.load_state_dict(checkpoint["model_state"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()
```

The current checks passed **61 tests**, with **one CUDA smoke test skipped**
because this machine has CPU-only PyTorch. On the desktop, that test runs
when CUDA is available. GPU execution has not been verified on this machine.

Implementation references: PyTorch's
[logit-based BCE loss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html)
and [checkpoint guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).
