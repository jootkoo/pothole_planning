"""Full encoder/decoder contract tests, independent of external image files."""

import pytest
import torch
from torch import nn

from models import PotNetDecoder, PotNetSegmentation


@pytest.fixture(scope="module", autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 2))
    yield
    torch.set_num_threads(previous)


def make_model():
    # Test choices; not settings attributed to the paper.
    return PotNetSegmentation(dropout_after_conv3=0.0, dropout_after_conv5=0.0)


@pytest.mark.parametrize("size", [(64, 64), (65, 67), (128, 192)])
def test_logits_match_original_input_dimensions(size):
    model = make_model().eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 3, *size))
    assert logits.shape == (1, 1, *size)
    assert torch.isfinite(logits).all()
    assert sum(isinstance(m, nn.Conv2d) for m in model.encoder.modules()) == 7
    assert sum(isinstance(m, nn.MaxPool2d) for m in model.encoder.modules()) == 6


def test_decoder_uses_six_skip_scales_and_returns_raw_logits():
    model = make_model().eval()
    seen = []
    handles = [stage.register_forward_pre_hook(lambda _m, inputs: seen.append(inputs[1].shape[1:]))
               for stage in model.decoder.stages]
    with torch.no_grad():
        # A fixed head outside [0,1] proves forward does not apply sigmoid.
        model.decoder.head.weight.zero_()
        model.decoder.head.bias.fill_(2.0)
        logits = model(torch.randn(1, 3, 64, 64))
    for handle in handles:
        handle.remove()
    assert seen == [(100, 2, 2), (150, 4, 4), (200, 8, 8),
                    (200, 16, 16), (250, 32, 32), (250, 64, 64)]
    assert torch.all(logits == 2.0)


def test_binary_loss_backpropagates_through_encoder_and_decoder():
    model = make_model().train()
    image = torch.randn(1, 3, 64, 64, requires_grad=True)
    target = torch.randint(0, 2, (1, 1, 64, 64)).float()
    loss = nn.BCEWithLogitsLoss()(model(image), target)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(image.grad).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.encoder.block1.conv.weight.grad.abs().sum() > 0
    assert model.decoder.head.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("channels", [(1, 2), (1, 2, 3, 4, 5, 0), (1, 2, 3, 4, 5, True)])
def test_invalid_decoder_channels_fail(channels):
    with pytest.raises(ValueError, match="six positive"):
        PotNetDecoder(channels)
