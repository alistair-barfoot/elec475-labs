# dataset.py
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

class CustomDataset(Dataset):
  def __init__(self, annotations_file, img_dir):
    self.img_labels = pd.read_csv(annotations_file, header=None, names=['filename', 'coordinates'])
    self.img_dir = img_dir

  def __len__(self):
    return len(self.img_labels)

  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])      
    image = Image.open(img_path)
     # Transform to ensure all images are [3, 227, 227]
    transform_to_227 = transforms.Compose([
        transforms.Resize((227, 227)),  # Resize to 227x227
        transforms.ToTensor()           # Convert to tensor [3, 227, 227]
    ])
    image = transform_to_227(image)
      
    label = self.img_labels.iloc[idx, 1]
    if self.transform:
        image = self.transform(image)
    if self.target_transform:
        label = self.target_transform(label)
    return image, label