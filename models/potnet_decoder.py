'''
HOW THIS DECODER WORKS

The encoder reads an RGB image and produces feature maps: grids of numbers
that can learn useful visual patterns. Its deepest map has a small height
and width because pooling repeatedly shrinks the image representation.
The decoder brings those maps back to the input image's height and width so
we can eventually label individual pixels as pothole or background.

There are three main pieces in this file:

1. _UpsampleAndFuse is one rebuilding stage. It enlarges a small feature map,
   joins it with an earlier encoder map (a "skip connection"), and applies
   two learned convolutions to combine the information. Earlier maps keep
   spatial detail that the small, deep maps alone cannot preserve.
2. PotNetDecoder runs six of those stages, from the smallest maps toward
   the full image size. A final 1x1 convolution produces one score per pixel.
3. PotNetSegmentation connects the existing encoder and this decoder into
   one trainable model: RGB image -> encoder -> decoder -> pixel scores.

Tensors use (N, C, H, W): batch size, channels, height, and width. A channel
is one feature map. The final output is (N, 1, H, W), with one raw score,
called a logit, for every pixel. A logit is not yet a probability. Training
uses these scores directly with BCEWithLogitsLoss. For inference, sigmoid
turns them into probabilities, and a chosen threshold turns them into a mask.

The sizes of the stages, skip choices, and convolutions below are our own
segmentation baseline. They are not a decoder reported in the PotNet paper.
The paper-based encoder keeps its seven convolutions and all six pools.
Both parts must be trained on image/mask pairs to learn useful predictions.
'''

import torch
# Tensor represents a grid of numbers. nn provides trainable network layers.
from torch import Tensor, nn
# F provides operations such as resizing that do not themselves have weights.
from torch.nn import functional as F

from .potnet_encoder import PotNetEncoder, PotNetFeatures


# Number of feature maps each rebuilding stage produces, deepest to shallowest.
# These are configurable project choices, not settings taken from the paper.
DEFAULT_DECODER_CHANNELS = (128, 96, 64, 48, 32, 16)


class _UpsampleAndFuse(nn.Module):
    """Enlarge features, join an encoder skip, and learn how to combine them."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        # Register this object as a PyTorch module so its weights can be trained.
        super().__init__()
        # in_channels: maps coming from the previous decoder stage.
        # skip_channels: maps arriving from an earlier encoder layer.
        # Concatenation stacks both sets, so the first convolution receives
        # in_channels + skip_channels maps rather than adding their values.
        self.refine = nn.Sequential(
            # A 3x3 filter learns from a pixel and its neighboring positions.
            # Padding 1 keeps height and width unchanged; out_channels chooses
            # how many new feature maps the convolution produces.
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            # ReLU replaces negative values with zero, adding nonlinearity.
            nn.ReLU(inplace=False),
            # A second convolution refines the combined features at this size.
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        # forward describes what happens to tensors during a model call.
        # shape[-2:] means (height, width). Match the actual skip size instead
        # of blindly doubling: pooling rounds down when image sizes are odd.
        # Bilinear interpolation estimates new values between existing pixels.
        # align_corners=False uses PyTorch's pixel-center resizing convention.
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        # dim=1 is the channel dimension in (N,C,H,W). Batch and spatial sizes
        # stay the same. The learned layers then mix these stacked channels.
        return self.refine(torch.cat((x, skip), dim=1))


class PotNetDecoder(nn.Module):
    """Six upsample/concatenate/refine stages and a one-channel logits head.

    Use conv7 at stride32, then conv5/4/3/2/1 at strides16/8/4/2/1.
    conv6 already feeds conv7 and shares its size, so it is not another scale.
    Channel widths, interpolation, refinement layers and head are project
    choices. See models/README.md; all parameters start from scratch.
    """
    # Names of the encoder maps used in order, from deepest to full resolution.
    # conv6 and conv7 have the same height/width: conv6 already feeds conv7.
    skip_keys = ("conv7", "conv5", "conv4", "conv3", "conv2", "conv1")

    def __init__(self, channels: tuple[int, ...] = DEFAULT_DECODER_CHANNELS):
        super().__init__()
        # Require one positive channel count for each of the six stages.
        if len(channels) != 6 or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in channels
        ):
            raise ValueError("Decoder channels must contain six positive integers.")
        # Channel counts of conv7, conv5, conv4, conv3, conv2, and conv1.
        skips = (100, 150, 200, 200, 250, 250)
        # The first stage receives the encoder's 100-channel final output.
        # Each following stage receives the previous decoder stage's output.
        inputs = (100,) + tuple(channels[:-1])
        # ModuleList registers every stage with PyTorch. This makes all stage
        # weights visible to model.parameters(), the optimizer, and checkpoints.
        self.stages = nn.ModuleList(
            _UpsampleAndFuse(cin, skip, cout)
            for cin, skip, cout in zip(inputs, skips, channels)
        )
        # A 1x1 convolution combines channels at each pixel into ONE pothole
        # score without changing height/width. This final layer is the "head".
        self.head = nn.Conv2d(channels[-1], 1, kernel_size=1)

    def forward(self, features: PotNetFeatures) -> Tensor:
        # Start with the smallest, deepest tensor returned by the encoder.
        x = features.output
        # Pair each decoder stage with its corresponding encoder skip map.
        for stage, key in zip(self.stages, self.skip_keys):
            x = stage(x, features.skips[key])
        # Return raw logits. BCEWithLogitsLoss includes sigmoid internally;
        # applying it here would make that loss receive the wrong kind of input.
        return self.head(x)


class PotNetSegmentation(nn.Module):
    """The complete model, containing both the encoder and the decoder."""
    def __init__(
        self, *, dropout_after_conv3: float, dropout_after_conv5: float,
        decoder_channels: tuple[int, ...] = DEFAULT_DECODER_CHANNELS,
        bn_eps: float = 1e-5, bn_momentum: float = 0.1,
    ):
        super().__init__()
        # Both dropout probabilities must be chosen explicitly because they
        # were not specified by the original encoder paper.
        self.encoder = PotNetEncoder(
            dropout_after_conv3=dropout_after_conv3,
            dropout_after_conv5=dropout_after_conv5,
            bn_eps=bn_eps, bn_momentum=bn_momentum,
        )
        # The two modules become children of one model, so a single optimizer
        # can update both parts during end-to-end training.
        self.decoder = PotNetDecoder(decoder_channels)

    def forward(self, images: Tensor) -> Tensor:
        # Ask for intermediate maps as well as the deepest encoder output.
        # Do not use no_grad() or detach(): decoder losses need to send
        # gradients back through these features to train the encoder too.
        features = self.encoder(images, return_features=True)
        return self.decoder(features)
