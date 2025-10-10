import torch
from torchvision import transforms
import os
import pandas as pd
from train import CustomDataset

# Define transform
transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor()
])

# Load dataset
dataset = CustomDataset(annotations_file='train_noses.txt', img_dir='images-original', transform=None)

for i in range(3):
                image, label = dataset[i]
                print(f"✅ Item {i}: shape={image.shape}, label={label}")
# Test sample (should not error)

