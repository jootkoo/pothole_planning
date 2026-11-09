# Encoder and segmentation checks

For actual optimizer updates, the new `train_segmentation.py` entry point has
`overfit` and `train` modes, supports CPU/CUDA, and saves weights and learning
history. Follow [training/README.md](../training/README.md) for commands and
desktop setup. The check scripts below remain verification tools.

For the new complete segmentation model and real-data checker, see [models/README.md](../models/README.md#verify-the-baseline). Run `python -m scripts.check_segmentation` with both dropout arguments from the project root; it loads the binary dataset, writes an annotation preview, and verifies one real-pair backward pass. The instructions below remain specific to the original encoder-only checker.

`check_potnet_encoder.py` demonstrates and verifies the outputs of the encoder implemented in [models/potnet_encoder.py](../models/potnet_encoder.py). Use it to inspect the feature sizes and parameter counts for a chosen input size.

## What the code does

1. Reads the command-line options and optional JSON configuration.
2. Requires explicit probabilities for the two dropout layers because the paper does not report them.
3. Sets the random seed and CPU thread count for this verification run.
4. Creates a randomly initialized `PotNetEncoder` and switches it to evaluation mode.
5. Generates a random RGB-shaped tensor, normally `(1, 3, 64, 64)`.
6. Runs the encoder with `torch.inference_mode()`, without computing gradients.
7. Checks and prints every pre-pooling skip shape, complete block output shape, and convolution parameter count.
8. Checks the total parameter/state counts and prints a final `PASS` message if everything agrees.

Evaluation mode disables dropout and makes BatchNorm use its running statistics. The dropout probabilities must still be supplied when constructing the model. The seed and CPU thread defaults are script choices, not reported paper settings.

This script uses random numbers instead of a real photograph. It does not train the encoder, perform a backward pass, load a dataset, save weights, produce a pothole mask, or measure recognition accuracy.

## How to run it

Run these PowerShell commands from the **project root**, the folder containing `models`, `scripts`, `tests`, and `.venv`. If your terminal is inside this `scripts` folder, first return to the project root:

```powershell
Set-Location ..
```

If the Python environment is not installed yet, follow [the main README's setup instructions](../README.md#install-and-verify).

Run the default 64×64 shape check:

```powershell
.\.venv\Scripts\python.exe -m scripts.check_potnet_encoder --dropout-after-conv3 0.5 --dropout-after-conv5 0.5
```

**Both `0.5` values are illustrative choices, not the authors' settings.** Use the `-m scripts.check_potnet_encoder` form from the project root so Python can import the shared `models` package.

Run with the supplied JSON config and explicit rate overrides:

```powershell
.\.venv\Scripts\python.exe -m scripts.check_potnet_encoder --config configs/potnet_paper.json --dropout-after-conv3 0.5 --dropout-after-conv5 0.5
```

The supplied config stores both rates as `null`. Either provide the overrides above or edit both config values to your chosen numeric probabilities. Once both are numeric, you can run:

```powershell
.\.venv\Scripts\python.exe -m scripts.check_potnet_encoder --config configs/potnet_paper.json
```

Command-line dropout values override the corresponding config values. The config also supports `bn_eps` and `bn_momentum`; their provenance is explained in [PAPER_EXACT_VS_ASSUMED.md](../PAPER_EXACT_VS_ASSUMED.md).

Try a different input size:

```powershell
.\.venv\Scripts\python.exe -m scripts.check_potnet_encoder --dropout-after-conv3 0.5 --dropout-after-conv5 0.5 --height 128 --width 192 --batch-size 1
```

This produces a final shape of `(1, 100, 2, 3)`. Support for other resolutions is a fully convolutional extension, not a claim about the paper's training resolution. Larger inputs increase memory use and computation.

Show all command-line options:

```powershell
.\.venv\Scripts\python.exe -m scripts.check_potnet_encoder --help
```

## Options

| Option | Meaning | Default |
| --- | --- | --- |
| `--config` | Path to a JSON object with encoder settings | No config |
| `--dropout-after-conv3` | Probability for dropout after Conv3's pool | Must be supplied here or in the config |
| `--dropout-after-conv5` | Probability for dropout after Conv5's pool | Must be supplied here or in the config |
| `--height` | Random input height; at least 64 | 64 |
| `--width` | Random input width; at least 64 | 64 |
| `--batch-size` | Number of random images; positive | 1 |
| `--threads` | Number of CPU threads; positive | 2 |
| `--seed` | Random seed for the verification run | 0 |

Both dropout probabilities must be finite and between 0 and 1, inclusive. Input dimensions need not be divisible by 64; the max pools round down.

## Expected output

For a batch of one 64×64 input image:

| Stage | Skip before pooling/dropout | Complete block output |
| --- | --- | --- |
| Block 1 | `(1,250,64,64)` | `(1,250,32,32)` |
| Block 2 | `(1,250,32,32)` | `(1,250,16,16)` |
| Block 3 | `(1,200,16,16)` | `(1,200,8,8)` |
| Block 4 | `(1,200,8,8)` | `(1,200,4,4)` |
| Block 5 | `(1,150,4,4)` | `(1,150,2,2)` |
| Block 6 | `(1,150,2,2)` | `(1,150,2,2)` |
| Block 7 | `(1,100,2,2)` | `(1,100,1,1)` |

The summary includes:

```text
Final output: (1, 100, 1, 1)
Convolution parameters: 3,800,050
Encoder trainable parameters (including BN affine): 3,801,050
BN running mean/variance values: 1,000
BN integer tracking counters: 3
Encoder Keras-style total (excluding integer counters): 3,802,050
Paper full classifier total, adding 202 head parameters: 3,802,252
PASS: all feature shapes and parameter counts match the documented interpretation.
```

The full-classifier count is a calculation for comparison with the paper; the script does not add a classification head. If a shape or count disagrees, an assertion fails instead of reporting `PASS`. Invalid options or config settings produce an error explaining what to correct.

For broader automated verification, including gradient flow and invalid-input handling, run the [unit tests](../tests/README.md).

## Real-data segmentation check

`check_segmentation.py` verifies the complete encoder/decoder with a real
training image and binary mask. Run from the project root:

```powershell
.\.venv\Scripts\python.exe -m scripts.check_segmentation --dropout-after-conv3 0.2 --dropout-after-conv5 0.3
```

These are explicit example rates, not paper settings. The command defaults to
the local Pothole Mix image dataset, a 128×128 input, batch1, and two CPU
threads. It checks filename pairing/counts for each existing split, writes a
ground-truth preview to `reports/potmix_training_preview.png`, and runs one
training-pair forward/backward check with `BCEWithLogitsLoss`.

Expect 3,340/496/504 samples, input `(1,3,128,128)`, target/logits
`(1,1,128,128)`, and a `PASS` message for shape, finite loss and gradients.
There is no optimizer update, checkpoint, epoch, or accuracy measurement.
Use `--help` for options. See [data/README.md](../data/README.md) for label
mapping and [models/README.md](../models/README.md) for decoder design.
