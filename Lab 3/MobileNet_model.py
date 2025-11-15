"""
adapted to use MobileNetV3-small as the encoder/backbone.

- Replaces the tiny custom backbone with a MobileNetV3-small encoder from torchvision.
- The encoder's `features` sequential is split into three parts (low, mid, high)
  dynamically (so it works across torchvision versions). This produces three
  feature taps for the decoder.
- The rest of the architecture (ASPP-lite context, lightweight decoder, dropout)
  

Notes:
- If `pretrained=True` the torchvision ImageNet weights are loaded (when available).
- The module inspects the backbone to discover output channel counts (so projections
  in the decoder are sized correctly).
- Inputs: RGB tensors (N,3,H,W). If using pretrained weights, apply ImageNet normalization.
"""
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary


# ---------------------------
# building blocks (unchanged)
# ---------------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation,
                                   groups=in_ch, bias=bias)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ---------------------------
# MobileNetV3-small backbone wrapper
# ---------------------------
class MobileNetV3SmallBackbone(nn.Module):
    """
    Loads torchvision.models.mobilenet_v3_small and splits its `features` Sequential
    into three sequential parts (low, mid, high). The split is done dynamically
    by dividing the number of feature blocks into three segments.

    Returns:
      low, mid, high feature maps (in that order) from forward().
    """
    def __init__(self, pretrained: bool = True, input_size: Tuple[int, int] = (224, 224)):
        super().__init__()
        # load mobilenet_v3_small; support both newer (weights=...) and older (pretrained=...) APIs
        try:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            mb = models.mobilenet_v3_small(weights=weights)
        except Exception:
            mb = models.mobilenet_v3_small(pretrained=pretrained)

        features = mb.features  # nn.Sequential
        n_blocks = len(features)
        # split into three parts (keep at least 1 layer per part)
        s1 = max(1, n_blocks // 3)
        s2 = max(s1 + 1, (2 * n_blocks) // 3)

        # create three sequential modules
        self.part1 = nn.Sequential(*features[:s1])        # low
        self.part2 = nn.Sequential(*features[s1:s2])      # mid
        self.part3 = nn.Sequential(*features[s2:])        # high

        # infer output channel sizes by forwarding a dummy tensor (CPU)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size[0], input_size[1])
            o1 = self.part1(dummy)
            o2 = self.part2(o1)
            o3 = self.part3(o2)
            self.out_channels = (o1.shape[1], o2.shape[1], o3.shape[1])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        low = self.part1(x)
        mid = self.part2(low)
        high = self.part3(mid)
        return low, mid, high


# ---------------------------
# ASPP-lite (unchanged)
# ---------------------------
class ASPPLite(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, rates: Tuple[int, int] = (6, 12), dropout: float = 0.1):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        r1, r2 = rates
        pad1 = r1
        pad2 = r2
        self.branch2 = nn.Sequential(
            DepthwiseSeparableConv(in_ch, out_ch, kernel_size=3, stride=1, padding=pad1, dilation=r1)
        )
        self.branch3 = nn.Sequential(
            DepthwiseSeparableConv(in_ch, out_ch, kernel_size=3, stride=1, padding=pad2, dilation=r2)
        )
        self.project = nn.Sequential(
            nn.Conv2d(3 * out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        cat = torch.cat([b1, b2, b3], dim=1)
        out = self.project(cat)
        out = self.dropout(out)
        return out


# ---------------------------
# decoder blocks (unchanged)
# ---------------------------
class DecoderConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------
# MobileNetV3-small encoder
# ---------------------------
class MBV3SmallSeg(nn.Module):
    """
    Option A segmentation model using MobileNetV3-small encoder.

    Args:
      num_classes: number of segmentation classes (Pascal VOC default 21).
      backbone_pretrained: whether to load torchvision ImageNet weights for MobileNetV3-small.
      aspp_out: channels produced by ASPP-lite on top of high-level features.
      decoder_channels: tuple specifying decoder channel sizes.
      input_size: tuple used when inferring backbone channels (default 224).
      dropout: dropout used in ASPP and before classifier.
    """
    def __init__(self,
                 num_classes: int = 21,
                 backbone_pretrained: bool = True,
                 aspp_out: int = 128,
                 decoder_channels: Tuple[int, int] = (128, 64),
                 input_size: Tuple[int, int] = (275,275),
                 dropout: float = 0.1):
        super().__init__()
        # load MobileNetV3-small backbone wrapper
        self.backbone = MobileNetV3SmallBackbone(pretrained=backbone_pretrained, input_size=input_size)
        low_ch, mid_ch, high_ch = self.backbone.out_channels

        # Context module (ASPP-lite) on high-level features
        self.context = ASPPLite(in_ch=high_ch, out_ch=aspp_out, rates=(6, 12), dropout=dropout)

        # Projectors for skip taps to match decoder channels (inferred from backbone)
        self.project_mid = nn.Sequential(
            nn.Conv2d(mid_ch, decoder_channels[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_channels[1]),
            nn.ReLU(inplace=True)
        )
        self.project_low = nn.Sequential(
            nn.Conv2d(low_ch, decoder_channels[1] // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_channels[1] // 2),
            nn.ReLU(inplace=True)
        )

        # Decoder stages
        self.decoder_reduce = nn.Sequential(
            nn.Conv2d(aspp_out, decoder_channels[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True)
        )
        self.decoder_block1 = DecoderConvBlock(decoder_channels[0] + decoder_channels[1], decoder_channels[1])
        low_proj_ch = decoder_channels[1] // 2
        self.decoder_block2 = DecoderConvBlock(decoder_channels[1] + low_proj_ch, decoder_channels[1] // 2)

        # Final classifier (dropout then 1x1 conv)
        self.classifier_dropout = nn.Dropout2d(p=dropout)
        self.classifier = nn.Conv2d(decoder_channels[1] // 2, num_classes, kernel_size=1)

        # initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for custom layers only, preserve pretrained backbone."""
        # Only initialize the non-backbone modules
        modules_to_init = [
            self.context,
            self.project_mid,
            self.project_low,
            self.decoder_reduce,
            self.decoder_block1,
            self.decoder_block2,
            self.classifier_dropout,
            self.classifier
        ]
        
        for module in modules_to_init:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    # Use smaller initialization for final classifier
                    if m == self.classifier:
                        nn.init.normal_(m.weight, 0, 0.01)  # Much smaller init for classifier
                    else:
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def _upsample(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[2], x.shape[3]

        # backbone -> low, mid, high
        low, mid, high = self.backbone(x)

        # context on high-level
        context = self.context(high)
        dec = self.decoder_reduce(context)

        # upsample to mid size and concat
        mid_size = mid.shape[2], mid.shape[3]
        dec = self._upsample(dec, size=mid_size)
        mid_proj = self.project_mid(mid)
        dec = torch.cat([dec, mid_proj], dim=1)
        dec = self.decoder_block1(dec)

        # upsample to low size and concat
        low_size = low.shape[2], low.shape[3]
        dec = self._upsample(dec, size=low_size)
        low_proj = self.project_low(low)
        dec = torch.cat([dec, low_proj], dim=1)
        dec = self.decoder_block2(dec)

        # upsample to original size and classify
        dec = self._upsample(dec, size=orig_size)
        dec = self.classifier_dropout(dec)
        logits = self.classifier(dec)
        return logits


# ---------------------------
# quick smoke test
# ---------------------------
if __name__ == '__main__':
    # Test the MobileNetV3-backed model with a dummy input
    model = MBV3SmallSeg(num_classes=21, backbone_pretrained=False, input_size=(256, 256))
    model.eval()
    inp = torch.randn(2, 3, 256, 256)
    out = model(inp)
    summary(model, input_size=(3, 256, 256))
    print("Input shape:", inp.shape)
    print("Output shape:", out.shape)  # expect (2, 21, 256, 256)