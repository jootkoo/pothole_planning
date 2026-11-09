"""Architecture, feature-contract, and gradient checks for the PotNet encoder."""

from collections import OrderedDict

import pytest
import torch
from torch import nn

from models.potnet_encoder import PotNetEncoder, PotNetFeatures


@pytest.fixture(scope="module", autouse=True)
def modest_cpu_thread_count():
    """Small test images are faster without a large CPU thread pool."""
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 2))
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def encoder():
    # Test values are explicit assumptions; the paper does not give these rates.
    return PotNetEncoder(dropout_after_conv3=0.2, dropout_after_conv5=0.3)


def test_seven_paper_convolutions_and_no_classifier(encoder):
    expected_channels = [(3, 250), (250, 250), (250, 200), (200, 200),
                         (200, 150), (150, 150), (150, 100)]
    convolutions = [module for module in encoder.modules() if isinstance(module, nn.Conv2d)]
    assert len(convolutions) == 7
    for index, (convolution, channels) in enumerate(zip(convolutions, expected_channels)):
        kernel = 5 if index < 3 else 3
        assert (convolution.in_channels, convolution.out_channels) == channels
        assert convolution.kernel_size == (kernel, kernel)
        assert convolution.stride == (1, 1)
        assert convolution.padding == (kernel // 2, kernel // 2)
        assert convolution.dilation == (1, 1)
        assert convolution.groups == 1
        assert convolution.bias is not None
    assert not any(isinstance(module, (nn.Linear, nn.Flatten, nn.Softmax))
                   for module in encoder.modules())


def test_pool_batchnorm_and_dropout_locations(encoder):
    for index in range(1, 8):
        block = getattr(encoder, f"block{index}")
        assert isinstance(block.activation, nn.ReLU)
        assert not block.activation.inplace
        if index in (4, 5, 6):
            assert isinstance(block.norm, nn.BatchNorm2d)
            assert block.norm.num_features == block.conv.out_channels
            assert block.norm.eps == 1e-5
            assert block.norm.momentum == 0.1
            assert block.norm.affine
            assert block.norm.track_running_stats
        else:
            assert block.norm is None
        if index in (1, 2, 3, 4, 5, 7):
            assert isinstance(block.pool, nn.MaxPool2d)
            assert block.pool.kernel_size == 2
            assert block.pool.stride == 2
            assert block.pool.padding == 0
            assert not block.pool.ceil_mode
        else:
            assert block.pool is None
        if index in (3, 5):
            assert type(block.dropout) is nn.Dropout
            assert block.dropout.p == (0.2 if index == 3 else 0.3)
            assert not block.dropout.inplace
        else:
            assert block.dropout is None


def test_runtime_operation_order_and_feature_capture_positions(encoder):
    """Hooks verify the operations actually execute in Table 2's order."""
    executed = []
    captured = {}
    handles = []
    expected = []
    for index in range(1, 8):
        block = getattr(encoder, f"block{index}")
        operations = ["conv", "activation"]
        if index in (4, 5, 6):
            operations.append("norm")
        if index != 6:
            operations.append("pool")
        if index in (3, 5):
            operations.append("dropout")
        for name in operations:
            label = f"block{index}.{name}"
            expected.append(label)

            def record(_module, _inputs, output, label=label):
                executed.append(label)
                captured[label] = output

            handles.append(getattr(block, name).register_forward_hook(record))
    try:
        encoder.eval()
        with torch.no_grad():
            result = encoder(torch.randn(1, 3, 64, 64), return_features=True)
    finally:
        for handle in handles:
            handle.remove()
    assert executed == expected
    for index in range(1, 8):
        skip_operation = "norm" if index in (4, 5, 6) else "activation"
        block_operation = "dropout" if index in (3, 5) else (
            "norm" if index == 6 else "pool"
        )
        assert result.skips[f"conv{index}"] is captured[f"block{index}.{skip_operation}"]
        assert result.blocks[f"block{index}"] is captured[f"block{index}.{block_operation}"]


@pytest.mark.parametrize("height,width", [(64, 64), (65, 67), (128, 192)])
def test_all_block_and_skip_shapes(encoder, height, width):
    encoder.eval()
    with torch.no_grad():
        result = encoder(torch.randn(1, 3, height, width), return_features=True)
    assert isinstance(result, PotNetFeatures)
    assert isinstance(result.blocks, OrderedDict)
    assert isinstance(result.skips, OrderedDict)
    assert list(result.blocks) == [f"block{index}" for index in range(1, 8)]
    assert list(result.skips) == [f"conv{index}" for index in range(1, 8)]
    channels = [250, 250, 200, 200, 150, 150, 100]
    # Six floor-mode pools, with no pool between convolutions six and seven.
    before_pool_divisors = [1, 2, 4, 8, 16, 32, 32]
    after_block_divisors = [2, 4, 8, 16, 32, 32, 64]
    for index, (channel_count, before, after) in enumerate(
        zip(channels, before_pool_divisors, after_block_divisors), start=1
    ):
        assert tuple(result.skips[f"conv{index}"].shape) == (
            1, channel_count, height // before, width // before
        )
        assert tuple(result.blocks[f"block{index}"].shape) == (
            1, channel_count, height // after, width // after
        )
    assert result.output is result.blocks["block7"]


def test_default_returns_final_tensor_and_eval_is_deterministic(encoder):
    encoder.eval()
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        output = encoder(image)
        features = encoder(image, return_features=True)
        repeated = encoder(image)
    assert isinstance(output, torch.Tensor)
    assert tuple(output.shape) == (1, 100, 1, 1)
    assert torch.equal(output, features.output)
    assert torch.equal(output, repeated)


def test_parameter_counts_distinguish_trainable_parameters_and_buffers(encoder):
    convolution_count = sum(parameter.numel() for module in encoder.modules()
                            if isinstance(module, nn.Conv2d) for parameter in module.parameters())
    assert convolution_count == 3_800_050
    assert sum(parameter.numel() for parameter in encoder.parameters()) == 3_801_050
    floating_buffers = [buffer for buffer in encoder.buffers() if buffer.is_floating_point()]
    integer_buffers = [buffer for buffer in encoder.buffers() if not buffer.is_floating_point()]
    assert sum(buffer.numel() for buffer in floating_buffers) == 1_000
    assert len(integer_buffers) == 3
    assert all(buffer.numel() == 1 for buffer in integer_buffers)
    assert sum(value.numel() for value in encoder.state_dict().values()) == 3_802_053


def test_backward_reaches_all_parameters_input_and_intermediate_features(encoder):
    encoder.train()
    image = torch.randn(1, 3, 64, 64, requires_grad=True)
    result = encoder(image, return_features=True)
    features = list(result.blocks.values()) + list(result.skips.values())
    for feature in features:
        assert feature.requires_grad
        feature.retain_grad()
    loss = result.output.square().mean()
    loss = loss + sum(feature.square().mean() for feature in features)
    loss.backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert image.grad.abs().sum() > 0
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for feature in features:
        assert feature.grad is not None
        assert torch.isfinite(feature.grad).all()


def test_dropout_obeys_train_and_eval_modes():
    # p=1 makes dropout behavior exact and avoids probabilistic assertions.
    encoder = PotNetEncoder(dropout_after_conv3=1.0, dropout_after_conv5=1.0)
    recorded = {}
    handles = []
    for index in (3, 5):
        def record(_module, inputs, output, index=index):
            recorded[index] = (inputs[0], output)
        handles.append(getattr(encoder, f"block{index}").dropout.register_forward_hook(record))
    try:
        image = torch.randn(1, 3, 64, 64)
        encoder.train()
        with torch.no_grad():
            encoder(image)
        assert all(torch.count_nonzero(output) == 0 for _, output in recorded.values())
        encoder.eval()
        with torch.no_grad():
            encoder(image)
        assert all(torch.equal(input_tensor, output) for input_tensor, output in recorded.values())
    finally:
        for handle in handles:
            handle.remove()


def test_dropout_probabilities_are_required():
    with pytest.raises(TypeError):
        PotNetEncoder()
    with pytest.raises(TypeError):
        PotNetEncoder(dropout_after_conv3=0.2)


@pytest.mark.parametrize("name", ["dropout_after_conv3", "dropout_after_conv5"])
@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_dropout_probabilities_are_rejected(name, value):
    arguments = {"dropout_after_conv3": 0.2, "dropout_after_conv5": 0.3, name: value}
    with pytest.raises(ValueError):
        PotNetEncoder(**arguments)


@pytest.mark.parametrize("name,value", [
    ("bn_eps", 0.0), ("bn_eps", -1.0), ("bn_eps", float("nan")),
    ("bn_momentum", -0.1), ("bn_momentum", 1.1), ("bn_momentum", float("inf")),
])
def test_invalid_batchnorm_settings_are_rejected(name, value):
    with pytest.raises(ValueError):
        PotNetEncoder(dropout_after_conv3=0.2, dropout_after_conv5=0.3, **{name: value})


@pytest.mark.parametrize("shape", [(3, 64, 64), (1, 1, 64, 64), (1, 4, 64, 64),
                                  (1, 3, 63, 64), (1, 3, 64, 63), (0, 3, 64, 64)])
def test_invalid_image_shapes_are_rejected(encoder, shape):
    with pytest.raises(ValueError):
        encoder(torch.randn(*shape))


def test_integer_images_are_rejected(encoder):
    with pytest.raises(TypeError):
        encoder(torch.zeros(1, 3, 64, 64, dtype=torch.uint8))


def test_batchnorm_hyperparameters_are_configurable():
    encoder = PotNetEncoder(dropout_after_conv3=0.0, dropout_after_conv5=0.0,
                           bn_eps=1e-3, bn_momentum=0.05)
    for index in (4, 5, 6):
        norm = getattr(encoder, f"block{index}").norm
        assert norm.eps == 1e-3
        assert norm.momentum == 0.05
