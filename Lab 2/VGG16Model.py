"""
VGG16-backed SnoutNet regression model (classifier-head replacement).

Loads torchvision.vgg16 (optionally pretrained) and replaces the final
classifier layer with a 2-output linear layer for regression (x,y).
"""
import torch
import torch.nn as nn
from torchvision import models

class SnoutNetVGG16_ClassifierHead(nn.Module):
    """
    Adaptation: load torchvision.vgg16 and replace the final classifier layer
    with nn.Linear(..., 2) for regression.
    If pretrained=True, inputs should be normalized with ImageNet mean/std.
    """
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        try:
            weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.vgg16(weights=weights)
        except Exception:
            # fallback for older torchvision versions
            self.model = models.vgg16(pretrained=pretrained)

        # Replace last classifier layer (1000 -> 2)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, 2)

        if freeze_backbone:
            for param in self.model.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
