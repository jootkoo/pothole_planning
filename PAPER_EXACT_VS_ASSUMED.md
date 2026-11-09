# Paper facts, inferences, and implementation choices

The later training loop's BCE/soft Dice loss, weights, threshold, learning
diagnostic, and hardware setup are project choices documented in
[training/README.md](training/README.md), rather than claims about the
original paper's segmentation method.

This document describes the original encoder reproduction. The subsequent project decoder is implemented in `models/potnet_decoder.py`; its new design choices are documented in [models/README.md](models/README.md). The encoder itself retains the topology described below. The binary dataset loader is documented in [data/README.md](data/README.md). Later references to the decoder as future work describe the original encoder milestone.

Source: Dewangan, D.K. and Sahu, S.P. (2021), *PotNet: Pothole detection for autonomous vehicle system using convolutional neural network*, Electronics Letters 57, 53–56, [DOI 10.1049/ell2.12062, publisher full text](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ell2.12062). Architecture references below refer to Table 2; ReLU and the stride/padding wording are in the accompanying prose.

The supplied project handoff is project context. Its future segmentation, dataset, stereo, and planning directions are not implemented in this encoder milestone. Where it labels a setting confirmed but the original table only supports an inference, this document uses the more precise provenance category.

| Setting | Provenance and implementation |
| --- | --- |
| Seven convolutions | **Reported:** output channels 250, 250, 200, 200, 150, 150, 100; kernels 5, 5, 5, 3, 3, 3, 3. Preserved without substitutions. |
| Input | **Reported:** 64×64×3. **Interface choice:** NCHW RGB tensors in PyTorch. Support for larger/rectangular inputs is an implementation extension, not a reported experiment. |
| Activation | **Reported:** ReLU in the prose, following each convolution in the table. ReLU is kept before BN where BN occurs. |
| BN placement | **Reported:** after activations 4, 5, and 6; before the pools following convolutions 4 and 5. |
| Pool placement and kernel | **Reported:** 2×2 max pools after convolutions 1, 2, 3, 4, 5, and 7. No pool after convolution 6. All six are preserved. |
| Dropout placement | **Reported:** immediately after the pools following convolutions 3 and 5. |
| Convolution stride | The prose states unit stride; stride 1 also agrees with the table's unchanged convolution spatial sizes. |
| Numeric convolution padding | **Inferred:** padding 2 for 5×5 and 1 for 3×3 at stride 1, to reproduce Table 2. The prose says padding and stride use “unit movement,” which conflicts with unchanged size for 5×5 kernels if padding is interpreted as 1. Table shapes take precedence here; padding 2 is not presented as an explicit author-supplied numeric setting. |
| Pool stride | **Inferred:** stride 2 reproduces each reported halving. The table lists pool kernel sizes but does not separately tabulate stride. |
| Convolution bias | **Inferred from reported counts:** each count equals `(Cin × kernel_area + 1) × Cout`, so `bias=True`. |
| Conv5 activation channels | **Apparent source typo:** row 16 says 100, whereas Conv5 and the following BN both say 150. Use 150; ReLU does not change channel count. No extra projection is invented. |
| Dropout probabilities | **Unreported:** the two constructor arguments are required separately and the config stores `null` until explicitly supplied. `0.5` in examples is an illustrative user choice, not a paper value. |
| Dropout type | **Implementation choice:** PyTorch elementwise `nn.Dropout`, not spatial/channel dropout. The paper does not distinguish variants. |
| BN numerical settings | **Unreported; PyTorch choices:** `eps=1e-5`, `momentum=0.1`, affine parameters enabled, running statistics tracked. Epsilon and momentum are configurable. PyTorch and Keras running-statistics conventions differ; no exact numerical cross-framework replication is claimed. |
| Other convolution settings | **Implementation choices:** ordinary ungrouped convolution, dilation 1, zero padding. No extra convolutions, dilation, residual links, or pretrained components. |
| Pool rounding and border behavior | **Implementation choices:** padding 0, dilation 1, `ceil_mode=False`. Thus odd dimensions round down, and H/W must each be at least 64 for all six pools. |
| Initialization and dtype | **Unreported:** retain PyTorch module initialization and its current default floating dtype (normally float32). Parameters are initialized from scratch. Inputs must match model dtype/device. No paper seed or pretrained checkpoint is assumed. |
| Image preprocessing | **Unreported:** no image preprocessing is built into this module; no normalization, scaling, or resizing recipe is attributed to the paper. |
| Feature export API | **Project interface choice:** post-block outputs and post-ReLU/BN pre-pool skips. The original classifier does not specify decoder skips. Maps remain attached to autograd when gradient tracking is enabled. |
| Final classifier | **Reported, excluded:** Flatten → Dense(2) → final activation/Softmax. The core encoder ends after the sixth pool; no reference classifier or decoder is added. |

## Parameter accounting

Convolution parameter counts from Table 2 are 19,000; 1,562,750; 1,250,200; 360,200; 270,150; 202,650; and 135,100. Their sum is **3,800,050**.

| Quantity | Count | Meaning |
| --- | ---: | --- |
| Seven convolutions | 3,800,050 | Trainable weights and biases |
| BN affine parameters | 1,000 | Trainable scale and offset for 200+150+150 channels |
| PyTorch encoder trainable total | **3,801,050** | Actual trainable encoder parameters |
| BN running means and variances | 1,000 | Non-trainable floating state |
| Encoder with Keras-style BN accounting | **3,802,050** | Trainable parameters plus running means/variances |
| Original Dense(100→2), calculation only | 202 | 100×2 weights plus 2 biases; no head implemented |
| Original full-classifier total | **3,802,252** | Matches the reported paper total |

PyTorch additionally stores three scalar integer `num_batches_tracked` buffers, one per BN module. Counting every element of the encoder `state_dict` therefore gives **3,802,053**, which includes counters and is not a trainable-weight count. The full paper total must not be compared directly with an encoder-only trainable count.

## Scope of reproducibility

This implementation reproduces the documented topology and Table 2 shapes while exposing unresolved numerical choices. It does not establish the authors' exact training setup, learned weights, classification accuracy, or segmentation performance. The unit tests and CLI verify architecture, shapes, parameter accounting, and differentiability. The future decoder and its training/evaluation remain separate work.
