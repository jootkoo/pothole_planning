"""PotNet's seven-convolution feature extractor, following Table 2.
Source: Dewangan and Sahu (2021), https://doi.org/10.1049/ell2.12062.
See PAPER_EXACT_VS_ASSUMED.md for unreported settings and table discrepancies.
This module produces features, not class scores or segmentation masks.
"""

from collections import OrderedDict #dictionary, preserves insertion order
from dataclasses import dataclass  #container for CNN output features
import math
from numbers import Real 
#Tensor - arrays, feature maps
#nn - Neural network module, contains pooling dropout etc
from torch import Tensor, nn 

#Main dataclass, frozen so attributes cannot be reassigned
@dataclass(frozen=True)
class PotNetFeatures:
    """Features from one forward pass, all still connected to autograd.
    output: Final tensor after the sixth and last max pool.
    blocks: block1 through block7, after each block's complete operation sequence.
    skips: conv1 through conv7, after ReLU/optional BatchNorm, before pooling
        and dropout. These preserve more spatial detail for a future decoder.
    """
    # The final tensor produced by the encoder AFTER ALL CONVO BLOCKS
    output: Tensor 

    #organizes feature maps in order in which the CNN produces, ORDERED
    #stores the OUTPUTS after each block finishes all of its operations, including pooling and dropout if applicable
    blocks: OrderedDict[str, Tensor]

    #stores feature maps captured BEFORE pooling and dropout
    #useful to create segmentation mask later, retaining earlier, higher-resolution features is valuable.
    skips: OrderedDict[str, Tensor]


#Checks whether the numerical setting is VALID
#value must be between min and max - dropout
def _validate_real(name: str, value: float, *, minimum: float, maximum: float) -> None:
    if (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not minimum <= value <= maximum):
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}].")


#CNN BLOCK
#inherits from pytorch's neural network class
class _PotNetBlock(nn.Module):
    """Conv -> ReLU -> optional BN -> optional pool -> optional dropout."""
    '''
    in_channels - Number of channels entering the convolution.
    out_channels - Number of channels produced by the convolution.
    kernel_size - Width and height of each square convolution filter.
    * - Requires the remaining arguments to be passed by keyword.
    batch_norm=false - batch normalization is disabled unless explicitly enabled.
    pool=True = Max pooling is enabled by default.
    dropout_probability=None - Dropout is disabled unless a probability is supplied. float | None means either a floating-point value or None.
    bn_eps - Numerical stability setting for BatchNorm.
    bn_momentum - Controls how BatchNorm updates its running statistics


    EX. 
    _PotNetBlock(
    in_channels=3,
    out_channels=250,
    kernel_size=5,
    batch_norm=False,
    pool=True,
    bn_eps=1e-5,
    bn_momentum=0.1
    )

    This creates a CNN block that takes an RGB image, 
    applies 250 convolution filters of size 5 x 5, uses ReLU, and then max pools the output.
    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        batch_norm: bool = False,
        pool: bool = True,
        dropout_probability: float | None = None,
        bn_eps: float,
        bn_momentum: float,) -> None:
        
        super().__init__()

        # Padding is inferred from Table 2's unchanged convolution output sizes.
        # The biases are required to match its convolution parameter counts.



        #CNN BLOCK 
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size,stride=1,padding=kernel_size // 2,bias=True,)
        self.activation = nn.ReLU(inplace=False)
        # Preserve the paper's unusual ReLU-before-BN order.
        self.norm = (nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_momentum) if batch_norm else None)
        self.pool = nn.MaxPool2d(2, stride=2) if pool else None
        # Elementwise Dropout is an explicit implementation choice; the paper
        # reports the locations but neither the probabilities nor the variant.
        self.dropout = (nn.Dropout(p=dropout_probability, inplace=False) if dropout_probability is not None else None)

    #forward pass
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.activation(self.conv(x))
        if self.norm is not None:
            x = self.norm(x)
        skip = x
        if self.pool is not None:
            x = self.pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x, skip


class PotNetEncoder(nn.Module):
    """RGB-only PotNet encoder with all seven convolutions and six pools.

    Both dropout probabilities are required because the paper omits them.
    bn_eps=1e-5 and bn_momentum=0.1 are PyTorch defaults, not reported PotNet
    settings. We retain PyTorch's default random initialization as well.

    Input: floating-point NCHW RGB tensor with H,W >= 64. The reference input
    is 64x64. Larger/rectangular inputs are a fully convolutional extension,
    not a claim about the paper's training data. Max pools round sizes down.

    forward(x) returns the final tensor (N,100,H//64,W//64).
    forward(x, return_features=True) returns PotNetFeatures instead.
    No input resizing, normalization, head, pretrained weights, or decoder is
    applied here. Tensors keep the caller's device and autograd connectivity.
    """

    def __init__(self,*,dropout_after_conv3: float,dropout_after_conv5: float,bn_eps: float = 1e-5,bn_momentum: float = 0.1,) -> None:
        super().__init__()
        _validate_real("dropout_after_conv3", dropout_after_conv3, minimum=0, maximum=1)
        _validate_real("dropout_after_conv5", dropout_after_conv5, minimum=0, maximum=1)
        if (isinstance(bn_eps, bool)or not isinstance(bn_eps, Real)or not math.isfinite(bn_eps)or bn_eps <= 0):
            raise ValueError("bn_eps must be positive and finite.")
        
        _validate_real("bn_momentum", bn_momentum, minimum=0, maximum=1)
        bn = {"bn_eps": bn_eps, "bn_momentum": bn_momentum}

        # Seven explicit blocks keep the published configuration easy to audit.
        self.block1 = _PotNetBlock(3, 250, 5, **bn)
        self.block2 = _PotNetBlock(250, 250, 5, **bn)
        self.block3 = _PotNetBlock(250, 200, 5, dropout_probability=dropout_after_conv3, **bn)
        self.block4 = _PotNetBlock(200, 200, 3, batch_norm=True, **bn)
        # Table 2's activation row says 100 here, but its convolution, BN and
        # parameter counts require 150. ReLU cannot change the channel count.
        self.block5 = _PotNetBlock(200, 150, 3, batch_norm=True,dropout_probability=dropout_after_conv5, **bn)
        self.block6 = _PotNetBlock(150, 150, 3, batch_norm=True, pool=False, **bn)
        self.block7 = _PotNetBlock(150, 100, 3, **bn)

    #forward pass
    def forward(self, x: Tensor, *, return_features: bool = False) -> Tensor | PotNetFeatures:
        if not isinstance(x, Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if x.ndim != 4:
            raise ValueError("Expected a four-dimensional NCHW RGB tensor.")
        if x.shape[1] != 3:
            raise ValueError("Expected 3 RGB channels in NCHW order.")
        if x.shape[0] < 1:
            raise ValueError("Batch size must be positive.")
        if min(x.shape[-2:]) < 64:
            raise ValueError("Input height and width must each be at least 64.")
        if not x.is_floating_point():
            raise TypeError("Input must be floating-point; preprocessing belongs to the caller.")

        blocks: OrderedDict[str, Tensor] = OrderedDict()
        skips: OrderedDict[str, Tensor] = OrderedDict()
        for index in range(1, 8):
            block = getattr(self, f"block{index}")
            x, skip = block(x)
            if return_features:
                # Do not detach or clone: future decoder losses must reach
                # these layers, and each block already avoids in-place edits.
                blocks[f"block{index}"] = x
                skips[f"conv{index}"] = skip
        if return_features:
            return PotNetFeatures(output=x, blocks=blocks, skips=skips)
        return x
