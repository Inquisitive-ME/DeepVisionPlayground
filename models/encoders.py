from enum import Enum, auto

import torch.nn as nn
import torchvision.models as models


class EncodeType(Enum):
    simple = auto()
    simple_bn = auto()
    simple_gn = auto()
    simple_gap = auto()
    simple_bn_gap = auto()
    simple_gn_gap = auto()
    resnet18 = auto()
    resnet18_spatial = auto()
    resnet34 = auto()
    resnet34_spatial = auto()


def _simple_block(in_ch: int, out_ch: int, norm: str) -> list[nn.Module]:
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
    ]
    if norm == "batch":
        layers.append(nn.BatchNorm2d(out_ch))
    elif norm == "group":
        # 8-group GroupNorm: per-sample, so train and eval behave identically
        # (no running-stats drift). All channel counts here are multiples of 8.
        layers.append(nn.GroupNorm(8, out_ch))
    elif norm != "none":
        raise ValueError(f"unknown norm: {norm}")
    layers.append(nn.ReLU(inplace=True))
    return layers


def _simple_stack(norm: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    layers.extend(_simple_block(3, 16, norm))      # 256 -> 128
    layers.extend(_simple_block(16, 32, norm))     # 128 -> 64
    layers.extend(_simple_block(32, 64, norm))     # 64 -> 32
    layers.extend(_simple_block(64, 128, norm))    # 32 -> 16
    return nn.Sequential(*layers)


def encoder(encoder_type: EncodeType) -> tuple[nn.Module, int]:
    if encoder_type is EncodeType.simple:
        # Original 4-conv stack. Keeps spatial 16x16x128 features so a FC
        # head can localize. Flatten included so the encoder always emits
        # a 2-D tensor; SimpleCenterNet's own Flatten then becomes a no-op.
        model: nn.Module = nn.Sequential(_simple_stack(norm="none"), nn.Flatten())
        features_out_size = 128 * 16 * 16
    elif encoder_type is EncodeType.simple_bn:
        # Same stack with BatchNorm after every conv. Converges quickly but
        # uses running stats that drift on fresh-data-per-epoch training, so
        # eval predictions diverge from train — NOT suitable for distribution-
        # shift studies. Prefer simple_gn for those.
        model = nn.Sequential(_simple_stack(norm="batch"), nn.Flatten())
        features_out_size = 128 * 16 * 16
    elif encoder_type is EncodeType.simple_gn:
        # Same stack with GroupNorm. Per-sample normalization, so train and
        # eval behave identically — the right default for the localization
        # tasks and for honest distribution-shift measurement.
        model = nn.Sequential(_simple_stack(norm="group"), nn.Flatten())
        features_out_size = 128 * 16 * 16
    elif encoder_type is EncodeType.simple_gap:
        # GAP'd to 128 features. Removes the 8M-parameter FC head you'd
        # otherwise get from flattening 16x16x128, but throws away the
        # spatial information a localizer needs. Only appropriate for
        # tasks where position doesn't matter (e.g. pure classification).
        model = nn.Sequential(
            _simple_stack(norm="none"),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        features_out_size = 128
    elif encoder_type is EncodeType.simple_bn_gap:
        model = nn.Sequential(
            _simple_stack(norm="batch"),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        features_out_size = 128
    elif encoder_type is EncodeType.simple_gn_gap:
        model = nn.Sequential(
            _simple_stack(norm="group"),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        features_out_size = 128
    elif encoder_type is EncodeType.resnet18:
        # Standard torchvision wiring: features go through AdaptiveAvgPool2d
        # before the fc head. We replace fc with Identity, so the encoder
        # outputs a (B, 512) GAP'd vector. That's the right thing for image
        # classification but POSITION-INVARIANT — i.e. cannot localize. Use
        # resnet18_spatial for localization tasks.
        rn18 = models.resnet18(weights=None)
        features_out_size = int(rn18.fc.in_features)
        rn18.fc = nn.Identity()
        model = rn18
    elif encoder_type is EncodeType.resnet18_spatial:
        # ResNet18 with the final avgpool + fc stripped, so the encoder emits
        # the (B, 512, 8, 8) feature map for 256x256 input. Flatten in the
        # head to localize. ~32k features after flatten.
        rn18 = models.resnet18(weights=None)
        model = nn.Sequential(
            rn18.conv1, rn18.bn1, rn18.relu, rn18.maxpool,
            rn18.layer1, rn18.layer2, rn18.layer3, rn18.layer4,
            nn.Flatten(),
        )
        features_out_size = 512 * 8 * 8
    elif encoder_type is EncodeType.resnet34:
        rn34 = models.resnet34(weights=None)
        features_out_size = int(rn34.fc.in_features)
        rn34.fc = nn.Identity()
        model = rn34
    elif encoder_type is EncodeType.resnet34_spatial:
        rn34 = models.resnet34(weights=None)
        model = nn.Sequential(
            rn34.conv1, rn34.bn1, rn34.relu, rn34.maxpool,
            rn34.layer1, rn34.layer2, rn34.layer3, rn34.layer4,
            nn.Flatten(),
        )
        features_out_size = 512 * 8 * 8
    else:
        raise ValueError(f"unknown encoder type: {encoder_type}")

    return model, features_out_size
