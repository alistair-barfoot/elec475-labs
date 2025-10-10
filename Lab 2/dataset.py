import os
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import Dataset
from model import snoutNet
import torch
from PIL import Image
import torchvision.transforms as transforms

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
      
      # Use PIL for image loading
      image = Image.open(img_path).convert("RGB")
      
      # Get original image dimensions for coordinate scaling
      original_width, original_height = image.size
      
      # Transform to ensure all images are [3, 227, 227]
      transform_to_227 = transforms.Compose([
          transforms.Resize((227, 227)),  # Resize to 227x227
          transforms.ToTensor()           # Convert to tensor [3, 227, 227]
      ])
      image = transform_to_227(image)
      
      # Transform coordinates based on image resizing
      label_str = self.img_labels.iloc[idx, 1]
      # Parse coordinates from string format "(x, y)"
      coords = label_str.strip('()').split(', ')
      original_x = float(coords[0])
      original_y = float(coords[1])
      
      # Scale coordinates to match the resized image (227x227)
      scaled_x = (original_x / original_width) * 227
      scaled_y = (original_y / original_height) * 227
      
      # Return scaled coordinates as a tensor
      label = torch.tensor([scaled_x, scaled_y], dtype=torch.float32)
      
      if self.transform:
          image = self.transform(image)
      if self.target_transform:
          label = self.target_transform(label)
      return image, label
  
# Example usage:
# dataset = CustomDataset(annotations_file='path/to/annotations.csv', img_dir='path/to/images', transform=your_transform)
def main():
  # Example usage:
  dataset = CustomDataset(annotations_file='train_noses.txt', img_dir='images', transform=None)
  print(f"Dataset length: {len(dataset)}")

  # Test just the first few items
  for i, (image, label) in enumerate(dataset):
    print(f"Item {i}: image shape = {image.shape}, label = {label}")
    if i >= 2:  # Only test first 3 items
      break

if __name__ == "__main__":
  main()