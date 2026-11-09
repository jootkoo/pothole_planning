# Encoder unit tests

`test_potnet_encoder.py` contains automated checks for the encoder implemented in [models/potnet_encoder.py](../models/potnet_encoder.py). Run these tests after changing model code to catch mistakes.

## What the code does

The tests create encoders and random image tensors, run the relevant operations, and use assertions to compare the results with the expected behavior. Pytest reports which checks pass or fail.

| Area | What is verified |
| --- | --- |
| Convolution layers | Exactly seven convolutions, with the expected channels, kernels, strides, padding, and biases; no classification head |
| Other layers | ReLU, BatchNorm, pooling, and dropout appear in their intended locations |
| Execution order | Runtime hooks confirm the operations actually run in the expected order and features are captured at the documented points |
| Output shapes | All block and skip maps have the expected sizes for 64×64, 65×67, and 128×192 inputs |
| Feature interface | Ordered feature mappings contain the correct keys, and the final output is the last block's tensor |
| Parameter accounting | Convolution parameters, BN affine parameters, running statistics, and integer counters are counted separately |
| Backward propagation | A temporary test loss produces finite gradients for the input, all model parameters, and the retained feature maps |
| Training/evaluation modes | Dropout follows `train()` and `eval()` behavior, and repeated evaluation outputs agree |
| Input validation | Invalid input shapes, integer images, missing/invalid dropout rates, and invalid BN settings are rejected; BN overrides are accepted |

The test fixture uses dropout rates of `0.2` and `0.3`. Another test uses `1.0` to make dropout behavior predictable. These are test inputs, not settings reported by the paper or recommendations for training.

This encoder test file contains **33 test cases**. Some functions run several times with different inputs. The complete suite now has **62 cases**, covering the loader, decoder, and training helpers; **61 passed and one CUDA test was skipped** on this CPU-only machine.

## How to run it

Run these PowerShell commands from the **project root**, the folder containing `models`, `scripts`, `tests`, and `.venv`. If your terminal is inside this `tests` folder, first return to the project root:

```powershell
Set-Location ..
```

Use the project's Python environment. If you have not installed it yet, follow [the main README's setup instructions](../README.md#install-and-verify).

Run this test file:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_potnet_encoder.py -q
```

Run all tests in the project:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Show the individual test names and results:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_potnet_encoder.py -v
```

Run only the backward-propagation test:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_potnet_encoder.py -k backward -v
```

## What the result means

A successful run of this encoder file reports `33 passed`; the current full suite reports `61 passed, 1 skipped` on this CPU-only machine. Elapsed time varies. If something fails, pytest identifies the test and shows the assertion or exception that failed. Review that output alongside your recent changes.

These tests verify implementation correctness. They use random tensors and a temporary loss for the gradient check; they do not load a pothole dataset, update model weights with an optimizer, save a trained model, or measure recognition accuracy.

For a readable table of encoder shapes and parameter counts, use the separate [shape-checking script](../scripts/README.md).

## Segmentation and loader tests

`test_segmentation.py` checks full-resolution logits for square, odd, and rectangular inputs, actual skip selection, absence of an output sigmoid, finite encoder/decoder gradients, and decoder width validation. It confirms the encoder still has seven convolutions and six pools.

`test_potmix.py` uses small synthetic files to check red/green/black mapping, nearest-neighbor binary resizing, pixel scaling, split/source pairing, uppercase image extensions, missing/mismatched pairs, EXIF handling, and invalid arguments. The downloaded dataset is not required for these unit tests.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_segmentation.py tests/test_potmix.py -q
```

For the separate real-data forward/backward check, follow [models/README.md](../models/README.md#verify-the-baseline).

## Training and CUDA checks

`test_training.py` checks the loss against known values, stable extreme logits,
global pixel metric aggregation, empty-mask handling, per-image overfit
criteria, real encoder/decoder weight updates, evaluation without BN/parameter
updates, checkpoint reload, and validation/test split isolation. A CLI check
confirms an explicit CUDA request fails before training if CUDA is unavailable.
The CUDA smoke test runs on supported hardware and is skipped locally.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_training.py -q
```

Separately from these unit tests, a four-image training-subset run passed its
learning criteria at update160 and the normal training/validation path was
smoke-tested on four training/two validation examples for one epoch.
Neither run establishes general segmentation quality. Read
[training/README.md](../training/README.md) for the artifacts and CUDA commands.
