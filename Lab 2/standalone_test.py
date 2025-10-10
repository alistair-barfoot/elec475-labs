import os
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
  def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    self.img_labels = pd.read_csv(annotations_file, header=None, names=['filename', 'coordinates'])
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform
  def __len__(self):
    return len(self.img_labels)
  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    with open(img_path, 'rb') as f:
      image = decode_image(f.read())
    label = self.img_labels.iloc[idx, 1]
    if self.transform:
      image = self.transform(image)
    if self.target_transform:
      label = self.target_transform(label)
    return image, label

# Test the dataset
print("Creating dataset...")
dataset = CustomDataset(annotations_file='train_noses.txt', img_dir='images', transform=None)
print(f"Dataset length: {len(dataset)}")

# Test first item
print("Loading first item...")
image, label = dataset[0]
print(f"Item 0: image shape = {image.shape}, label = {label}")

# Test second item
print("Loading second item...")
image, label = dataset[1]
print(f"Item 1: image shape = {image.shape}, label = {label}")

# Test final item
print("Loading final item...")
image, label = dataset[len(dataset)-1]
print(f"Item {len(dataset)-1}: image shape = {image.shape}, label = {label}")

print("Dataset test completed successfully!")