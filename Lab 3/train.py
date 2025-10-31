from torchvision.models.segmentation import fcn_resnet50
import torch
import torch.nn as nn
import torch.optim as optim
# import pascal voc 2012 dataset
from torchsummary import summary
from torchvision.datasets import VOCSegmentation

def main():
    model = fcn_resnet50(pretrained=True, progress=True, num_classes=21, aux_loss=None)
    summary(model)

if __name__ == "__main__":
    main()
