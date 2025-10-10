import torch
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from train import PetNoseDataset

# Read the first annotation to get a filename and coordinates
with open("train_noses.txt", "r") as f:
    first_line = f.readline()
filename, coords = first_line.strip().split(",", 1)
filename = filename.strip().strip('"')
img_dir = "test_images"
os.makedirs(img_dir, exist_ok=True)

# Create a dummy image for the first entry
dummy_img = Image.new("RGB", (227, 227), color=(123, 222, 111))
dummy_img.save(os.path.join(img_dir, filename))

# Define transform
transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor()
])

# Load dataset
dataset = PetNoseDataset("train_noses.txt", img_dir, transform=transform)

# Test sample (should not error)
img, target = dataset[0]
print("✅ Image shape:", img.shape)   # should be [3, 227, 227]
print("✅ Target:", target)           # should be normalized [x/w, y/h]

assert isinstance(img, torch.Tensor)
assert isinstance(target, torch.Tensor)
assert img.shape == (3, 227, 227)
print("✅ PetNoseDataset test with train_noses.txt passed!")