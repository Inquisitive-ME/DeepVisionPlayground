from enum import Enum, auto

import torch.nn as nn
import torchvision.models as models


class EncodeType(Enum):
    simple = auto()
    simple_bn = auto()
    simple_gap = auto()
    simple_bn_gap = auto()
    resnet18 = auto()
    resnet18_spatial = auto()
    resnet34 = auto()
    resnet34_spatial = auto()


def _simple_block(in_ch: int, out_ch: int, with_bn: bool) -> list[nn.Module]:
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
    ]
    if with_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.ReLU(inplace=True))
    return layers


def _simple_stack(with_bn: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    layers.extend(_simple_block(3, 16, with_bn))      # 256 -> 128
    layers.extend(_simple_block(16, 32, with_bn))     # 128 -> 64
    layers.extend(_simple_block(32, 64, with_bn))     # 64 -> 32
    layers.extend(_simple_block(64, 128, with_bn))    # 32 -> 16
    return nn.Sequential(*layers)


def encoder(encoder_type: EncodeType) -> tuple[nn.Module, int]:
    if encoder_type is EncodeType.simple:
        # Original 4-conv stack. Keeps spatial 16x16x128 features so a FC
        # head can localize, but is slow to converge without BN.
        model: nn.Module = _simple_stack(with_bn=False)
        features_out_size = 128 * 16 * 16
    elif encoder_type is EncodeType.simple_bn:
        # Same stack with BatchNorm after every conv. Typically 5-10x
        # faster to convergence than the no-BN variant on these synthetic
        # tasks, at the cost of a tiny number of extra parameters.
        model = _simple_stack(with_bn=True)
        features_out_size = 128 * 16 * 16
    elif encoder_type is EncodeType.simple_gap:
        # GAP'd to 128 features. Removes the 8M-parameter FC head you'd
        # otherwise get from flattening 16x16x128, but throws away the
        # spatial information a localizer needs. Only appropriate for
        # tasks where position doesn't matter (e.g. pure classification).
        model = nn.Sequential(
            _simple_stack(with_bn=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        features_out_size = 128
    elif encoder_type is EncodeType.simple_bn_gap:
        model = nn.Sequential(
            _simple_stack(with_bn=True),
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
