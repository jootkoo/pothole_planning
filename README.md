# PotNet encoder and segmentation baseline

The seven-convolution encoder follows Table 2 of Dewangan and Sahu (2021), [*PotNet: Pothole detection for autonomous vehicle system using convolutional neural network*](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ell2.12062), retaining all six original pools. A project-designed decoder converts its features into one raw pothole score per pixel. The binary Pothole Mix loader, training/validation loop, pixel metrics, and checkpoint saving are implemented. Stereo processing and planning remain later milestones.

## Current segmentation milestone

- [training/README.md](training/README.md): training commands, completed learning diagnostic, saved weights, and desktop CUDA setup.
- [models/README.md](models/README.md): decoder design choices, usage, and commands.
- [data/README.md](data/README.md): verified dataset layout, label mapping, preprocessing, and split counts.
- Tests: **61 passed, 1 CUDA test skipped** on this CPU-only machine. The training tests were also rerun after the final preview/progress changes: **11 passed, 1 CUDA test skipped**.
- A four-image 64×64 training-subset diagnostic passed at update160: IoU0.918848, Dice0.957708, loss reduced from1.675878 to0.055940. Each positive image reached IoU>=0.90 and both negative masks had zero false positives. Saved weights reload and reproduce the result.
- These numbers measure memorization of four training images. A full dataset experiment and final held-out testing have not run. The existing `check_segmentation` command still performs a forward/backward check without optimizer updates.

## Training and the desktop GPU

Start with a fresh tiny-set learning check:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_segmentation --mode overfit --device cpu --dropout-after-conv3 0 --dropout-after-conv5 0 --height 64 --width 64 --batch-size 4 --steps 200 --output runs/overfit_check
```

Choose a new output directory for each run. Completed diagnostic artifacts are in `runs/overfit_4x64_v1/`, including `before.png`, `after.png`, `history.csv`, `best.pt`, `last.pt`, and `summary.json`. These weights are a sanity-check result rather than a generally trained pothole detector.

The desktop's graphics card will be checked later. A Ryzen5 describes its CPU, and 32–64GB describes system RAM; CUDA needs a supported NVIDIA GPU with its own VRAM. Create a new CUDA-enabled environment using [the desktop guide](training/README.md#2-set-up-the-desktop-for-gpu-training), rather than copying this CPU-only `.venv`. The CLI supports `--device cuda`, `--device cpu`, and automatic selection. GPU execution has not been tested locally.

Normal training updates on the official training split and selects a checkpoint on the separate validation split. It never opens the testing split. An example desktop command, after verifying CUDA:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_segmentation --mode train --device cuda --dropout-after-conv3 0.2 --dropout-after-conv5 0.3 --height 128 --width 128 --batch-size 4 --epochs 20 --output runs/potnet_baseline
```

The example hyperparameters are project choices, not prescribed segmentation settings from the paper. Choose the training resolution and batch size after checking desktop VRAM.

## Check the model without training

Run the real-pair checker from the project root; the dropout rates are example choices:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m scripts.check_segmentation --dropout-after-conv3 0.2 --dropout-after-conv5 0.3
```

The dataset defaults to `potmix-dataset/pothole-mix-v1.0-20220526/pothole-mix`. The checker writes `reports/potmix_training_preview.png` showing annotations and binary targets. Red pothole pixels become 1; green cracks and black background become 0.

The source leaves several settings unresolved. [PAPER_EXACT_VS_ASSUMED.md](PAPER_EXACT_VS_ASSUMED.md) records the published facts, shape-based inferences, apparent table typo, and implementation choices. In particular, neither dropout probability has a default: callers must choose both explicitly. No pretrained model or weights are used.

## What it does

The encoder takes an RGB image through seven convolutional layers. Each layer applies adjustable filters and produces numerical **feature maps**. During training, these filters can learn patterns useful for recognizing potholes, such as edges, textures, and shapes.

Pooling gradually reduces the maps' height and width. For the paper's 64×64 input, the final output is 100 channels at 1×1 resolution. For a batch of one image, the tensor shape is `(1, 100, 1, 1)`.

Intermediate maps retain spatial detail that the final 1×1 output cannot preserve on its own. The segmentation decoder combines deeper features with these earlier maps through skip connections to predict a label for each image pixel.

Each new model starts with randomly initialized weights. The saved four-image diagnostic shows that encoder and decoder can learn together, but it does not establish general pothole recognition. The encoder alone returns features; the complete encoder/decoder returns per-pixel logits.

Other operations follow the published ordering:

- ReLU adds the nonlinearity needed to learn complex patterns.
- Batch normalization normalizes activations using batch or running statistics.
- Dropout randomly zeros some activations during training as regularization. It is disabled in evaluation mode.

## What each file is for

| File | Purpose | When to use it |
| --- | --- | --- |
| `models/potnet_encoder.py` | The actual encoder and its feature-output interface | Import `PotNetEncoder` when building your segmentation model |
| `models/potnet_decoder.py` | Project decoder and complete `PotNetSegmentation` model | Produce raw per-pixel logits from RGB inputs |
| `data/potmix.py` | Binary pothole image/mask loader | Load the existing training, validation, and testing splits |
| `scripts/check_segmentation.py` | Real-pair forward/backward check and annotation preview | Verify the loader and complete model together |
| `scripts/train_segmentation.py` | Tiny-set diagnostic or normal training/validation CLI | Run optimizer updates and save results |
| `training/segmentation.py` | Loss, metrics, optimization, evaluation and checkpoint helpers | Reusable training components |
| `scripts/check_potnet_encoder.py` | Passes random input through the encoder and prints/checks output sizes and parameter counts | Run it to see and verify the encoder's outputs |
| `tests/test_potnet_encoder.py` | 33 automated checks for layer configuration, execution order, shapes, gradients, and validation | Run after code changes to catch mistakes |
| `configs/potnet_paper.json` | Explicit numerical settings; the two unreported dropout rates start as `null` | Record your chosen settings and pass the config to the checker |
| `PAPER_EXACT_VS_ASSUMED.md` | Separates paper facts, shape-based inferences, and implementation choices | Review the architecture's provenance and reproduction limits |
| `README.md` | Explanation, setup, architecture, API examples, and next project steps | Start here |
| `requirements.txt` | Python dependencies | Install them in your environment |

Both the script and the tests import the same encoder from `models`. Neither is a training program. The `__init__.py` files organize Python imports; `__pycache__` folders are generated automatically.

Folder-specific guides explain the code and its commands: [unit tests](tests/README.md) and [shape/parameter checker](scripts/README.md).

The package contains source code, tests, and documentation. Install its Python dependencies on the machine where you will run it; the ZIP does not contain the original machine's virtual environment, trained weights, or datasets.

## Install and verify

Open PowerShell in the project directory containing this README. If using the ZIP, extract it and open the extracted `PotNetEncoder` folder first. Python 3.11 is a suitable environment choice; it is not a paper setting.

The following commands install the **CPU build** for local development. For the GPU desktop, use the [CUDA setup instructions](training/README.md#2-set-up-the-desktop-for-gpu-training) instead of installing the CPU wheel.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "torch>=2.2,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.check_potnet_encoder --dropout-after-conv3 0.5 --dropout-after-conv5 0.5
```

**The two `0.5` values above are examples chosen for this invocation, not values reported by the authors.** The shape CLI uses random input and does not train or evaluate pothole recognition. Its optional `--height`, `--width`, `--batch-size`, `--threads`, and `--seed` arguments control the verification run. A small batch is appropriate; the paper's training batch size is not a smoke-test requirement.

The config intentionally keeps both dropout probabilities `null`. Fill them with your explicit choices, or supply CLI overrides:

Requirements are `torch>=2.2,<3`, `numpy>=1.26,<3`, `Pillow>=10,<13`, and `pytest>=8,<10`. NumPy supports array interoperability and Pillow loads/resizes dataset images. The encoder itself uses only PyTorch. Passing these checks does not imply recognition accuracy or segmentation performance.

The original encoder-only milestone was validated locally on CPU with Python 3.11, PyTorch 2.14.1+cpu, NumPy 2.4.6, and pytest 9.1.1 (the current suite has 61 passing cases and one locally skipped CUDA case):

```text
33 passed in 4.23s
PASS: all feature shapes and parameter counts match the documented interpretation.
No broken requirements found.
```

The suite checks the actual execution order, layer settings, feature capture positions, 64×64/65×67/128×192 shapes, parameter/state counts, dropout modes, input validation, and backward propagation through the input, all parameters, and intermediate features.

## Architecture and output shapes

Every convolution has stride 1 and a bias, followed by ReLU. Same-size convolution padding is inferred from Table 2's output sizes: 2 for 5×5 kernels and 1 for 3×3 kernels. Each listed pool is 2×2 with inferred stride 2. BN follows ReLU in blocks 4, 5, and 6. This preserves the source ordering, including the absence of pooling between convolutions 6 and 7.

For input `(N, 3, 64, 64)`:

| Block | Convolution channels; kernel | Operations after convolution | Exported `blocks` shape | Exported `skips` shape |
| --- | --- | --- | --- | --- |
| 1 | 3→250; 5×5 | ReLU → pool | `(N,250,32,32)` | `(N,250,64,64)` |
| 2 | 250→250; 5×5 | ReLU → pool | `(N,250,16,16)` | `(N,250,32,32)` |
| 3 | 250→200; 5×5 | ReLU → pool → dropout | `(N,200,8,8)` | `(N,200,16,16)` |
| 4 | 200→200; 3×3 | ReLU → BN → pool | `(N,200,4,4)` | `(N,200,8,8)` |
| 5 | 200→150; 3×3 | ReLU → BN → pool → dropout | `(N,150,2,2)` | `(N,150,4,4)` |
| 6 | 150→150; 3×3 | ReLU → BN | `(N,150,2,2)` | `(N,150,2,2)` |
| 7 | 150→100; 3×3 | ReLU → pool | `(N,100,1,1)` | `(N,100,2,2)` |

The final tensor contains 100 channels at nominal stride 64. For supported inputs with height and width at least 64, floor-mode pooling gives spatial dimensions `(H // 64, W // 64)`. Other resolutions and rectangular inputs are implementation support, not evidence that the paper trained at those sizes. There is no resizing, normalization, or RGB conversion inside the model; supply floating NCHW RGB tensors with the model's dtype and device.

The seven convolutions contain **3,800,050** parameters. PyTorch BN adds 1,000 trainable affine parameters, making **3,801,050 trainable encoder parameters**. The paper's **3,802,252** total belongs to its complete classifier and includes BN running statistics; the accounting is explained in the provenance document.

## Python interface

```python
import torch
from models.potnet_encoder import PotNetEncoder

# These probabilities are explicit example choices; the paper does not supply them.
encoder = PotNetEncoder(dropout_after_conv3=0.5, dropout_after_conv5=0.5)
encoder.eval()
images = torch.randn(1, 3, 64, 64)

with torch.no_grad():
    final_tensor = encoder(images)  # Tensor: (1, 100, 1, 1)
    features = encoder(images, return_features=True)

assert torch.equal(final_tensor, features.output)
assert features.blocks["block7"] is features.output
print(features.output.shape)           # (1, 100, 1, 1): final encoder output
print(features.blocks["block1"].shape)  # (1, 250, 32, 32): complete block output
print(features.skips["conv1"].shape)    # (1, 250, 64, 64): full-resolution skip
print(features.skips["conv7"].shape)    # (1, 100, 2, 2): deepest pre-pool map
```

`return_features=True` returns a frozen `PotNetFeatures` dataclass with `.output`, `.blocks`, and `.skips`. The ordered `blocks` mapping contains `block1` through `block7` **after every operation in each block**, including pooling and dropout. The ordered `skips` mapping contains `conv1` through `conv7` **after ReLU and optional BN, before pooling and dropout**. These exports add no network operations. Outside a `no_grad()` context, tensors retain their computation graphs for future end-to-end training. `train()` and `eval()` retain normal PyTorch dropout and BN behavior.

The training loop uses normal gradient tracking instead of `no_grad()`. Exported maps stay connected to the encoder's computation graph, so the decoder's loss updates the encoder too.

## How the decoder connects

The new decoder starts from `features.output`, progressively upsamples, and merges `features.skips` to recover spatial detail. It selects conv7, conv5, conv4, conv3, conv2, and conv1 at strides32/16/8/4/2/1. Convolutions 6 and 7 share a resolution, so conv6 is not treated as another scale. Upsampling matches actual skip dimensions, including for odd inputs. See [models/README.md](models/README.md) for the complete design.

A 1×1 bottleneck at the paper's 64×64 input is insufficient by itself to preserve fine pothole boundaries; early maps provide the decoder with spatial detail. The complete model returns raw logits; inference can apply sigmoid and a chosen threshold to produce a mask after training. Larger inputs retain this encoder topology but increase cost: at fixed batch size, changing 64×64 to 256×256 increases early feature memory and convolution work roughly 16-fold, and 512×512 roughly 64-fold. The 250-channel early layers make small batches useful during development.

## Next steps toward segmentation

1. **Decide your experiment settings.** Record both dropout rates and the intended input resolution. Retain this encoder as the baseline. Larger inputs cost more memory and compute; changing pooling would be a separate architecture adaptation.
2. **Prepare RGB image/mask pairs.** Following the existing project plan, inspect the intended segmentation dataset's documentation and mask class values. Map pothole pixels to 1 and the other pixels to 0. Keep training, validation, and test groups separate. Apply matching spatial transforms to images and masks, using nearest-neighbor interpolation for masks.
3. **Review the implemented decoder baseline.** Its new design is documented separately from the paper-based encoder. Keep it as the initial segmentation baseline for comparison with later changes.
4. **Run the implemented trainer on the desktop.** Verify CUDA and repeat the tiny-set diagnostic, then train using the official validation split. Review loss curves and predicted masks before extending training. The random-input shape checker remains a separate tool.
5. **Evaluate on held-out images.** Measure pothole segmentation quality using metrics such as IoU and Dice, inspect predicted masks visually, and measure inference time on the intended hardware.
6. **Integrate stereo depth later.** Once useful masks are available, align them with calibrated stereo depth for the separate 3D stage. RGB encoder training itself does not require stereo data. Severity estimation and vehicle planning remain later milestones.
