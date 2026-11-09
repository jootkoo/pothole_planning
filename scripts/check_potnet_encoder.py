"""Verify feature shapes and parameter accounting on a random RGB tensor.

Run from the repository root with: python -m scripts.check_potnet_encoder
Both dropout probabilities must be supplied explicitly or in --config.
"""

import argparse
import json
from pathlib import Path

import torch

from models.potnet_encoder import PotNetEncoder, PotNetFeatures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON encoder settings; null rates need overrides")
    parser.add_argument("--dropout-after-conv3", type=float)
    parser.add_argument("--dropout-after-conv5", type=float)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2, help="CPU threads for this check (default: 2)")
    parser.add_argument("--seed", type=int, default=0, help="Smoke-test seed, not a paper setting")
    args = parser.parse_args()

    config = {}
    if args.config is not None:
        try:
            config = json.loads(args.config.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            parser.error(f"Cannot load config: {exc}")
        allowed = {"dropout_after_conv3", "dropout_after_conv5", "bn_eps", "bn_momentum"}
        if not isinstance(config, dict) or set(config) - allowed:
            parser.error("Config must be an object containing only the four documented encoder settings.")
    for name in ("dropout_after_conv3", "dropout_after_conv5"):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
        if config.get(name) is None:
            parser.error(f"Supply --{name.replace('_', '-')} or a numeric {name} in --config; the paper omits it.")
    if args.threads < 1 or args.batch_size < 1 or min(args.height, args.width) < 64:
        parser.error("Use positive threads/batch size and height/width >= 64.")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    try:
        model = PotNetEncoder(**config).eval()
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    x = torch.randn(args.batch_size, 3, args.height, args.width)
    with torch.inference_mode():
        features = model(x, return_features=True)
    assert isinstance(features, PotNetFeatures)

    print(f"PyTorch {torch.__version__}; CPU; eval mode; random input (no training)")
    print(f"Input: {tuple(x.shape)}")
    print("User-selected dropout probabilities (not reported in paper): "
          f"{config['dropout_after_conv3']}, {config['dropout_after_conv5']}")
    if (args.height, args.width) != (64, 64):
        print("Non-paper input resolution: fully convolutional extension with all six pools retained.")
    print("stage   | skip before pool/dropout | complete block output  | conv parameters")
    channels = (250, 250, 200, 200, 150, 150, 100)
    skip_divisors = (1, 2, 4, 8, 16, 32, 32)
    block_divisors = (2, 4, 8, 16, 32, 32, 64)
    conv_parameters = 0
    for index, (channel, skip_divisor, block_divisor) in enumerate(
        zip(channels, skip_divisors, block_divisors), start=1
    ):
        skip = features.skips[f"conv{index}"]
        block = features.blocks[f"block{index}"]
        assert tuple(skip.shape) == (
            args.batch_size, channel, args.height // skip_divisor, args.width // skip_divisor
        )
        assert tuple(block.shape) == (
            args.batch_size, channel, args.height // block_divisor, args.width // block_divisor
        )
        count = sum(p.numel() for p in getattr(model, f"block{index}").conv.parameters())
        conv_parameters += count
        print(f"block{index}  | {str(tuple(skip.shape)):<24} | {str(tuple(block.shape)):<22} | {count:,}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bn_running_state = sum(b.numel() for b in model.buffers() if b.is_floating_point())
    integer_counters = sum(b.numel() for b in model.buffers() if not b.is_floating_point())
    assert conv_parameters == 3_800_050
    assert trainable == 3_801_050
    assert bn_running_state == 1_000
    assert integer_counters == 3
    assert features.output is features.blocks["block7"]
    print(f"Final output: {tuple(features.output.shape)}")
    print(f"Convolution parameters: {conv_parameters:,}")
    print(f"Encoder trainable parameters (including BN affine): {trainable:,}")
    print(f"BN running mean/variance values: {bn_running_state:,}")
    print(f"BN integer tracking counters: {integer_counters}")
    print(f"Encoder Keras-style total (excluding integer counters): {trainable + bn_running_state:,}")
    print(f"Paper full classifier total, adding 202 head parameters: {trainable + bn_running_state + 202:,}")
    print("PASS: all feature shapes and parameter counts match the documented interpretation.")


if __name__ == "__main__":
    main()
