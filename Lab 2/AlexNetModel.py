
import torch
import torch.nn as nn
from torchvision import models

class SnoutNetAlexNet_ClassifierHead(nn.Module):
    """
    Simple adaptation: load torchvision.alexnet and replace the last classifier layer
    with nn.Linear(..., 2) for regression.
    Inputs should be normalized with ImageNet mean/std when pretrained=True.
    """
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        try:
            weights = models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.alexnet(weights=weights)
        except Exception:
            # older torchvision API
            self.model = models.alexnet(pretrained=pretrained)

        # replace last classifier layer (1000 -> 2)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, 2)

        if freeze_backbone:
            for param in self.model.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
