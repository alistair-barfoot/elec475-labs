import os
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import Dataset
from model import snoutNet
import torch
from PIL import Image
import torchvision.transforms as transforms

class CustomDataset(Dataset):
  def __init__(self, annotations_file, img_dir, transform):
      self.img_labels = pd.read_csv(annotations_file, header=None, names=['filename', 'coordinates'])
      self.img_dir = img_dir
      self.transform = transform

  def __len__(self):
      return len(self.img_labels)

  def __getitem__(self, idx):
      img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
      
      # Use PIL for image loading
      image = Image.open(img_path).convert("RGB")
      
      # Get original image dimensions for coordinate scaling
      original_width, original_height = image.size
      
      # Transform to ensure all images are [3, 227, 227]
      image = self.transform(image)
      
      # Transform coordinates based on image resizing
      label_str = self.img_labels.iloc[idx, 1]
      # Parse coordinates from string format "(x, y)"
      coords = label_str.strip('()').split(', ')
      original_x = float(coords[0])
      original_y = float(coords[1])
      
      # Scale coordinates to match the resized image (227x227)
      scaled_x = (original_x / original_width) * 227
      scaled_y = (original_y / original_height) * 227

      # Check whether the transform pipeline contains a RandomHorizontalFlip and, if so, decide/apply a flip to coordinates
      flip_applied = False
      transforms_list = []
      if isinstance(self.transform, transforms.Compose):
        transforms_list = self.transform.transforms
      elif isinstance(self.transform, (list, tuple)):
        transforms_list = list(self.transform)
      else:
        transforms_list = [self.transform]

      for t in transforms_list:
        if t.__class__.__name__ == "RandomHorizontalFlip":
          flip_applied = True
          break

      if flip_applied:
        # If the image was flipped horizontally, mirror the x coordinate on the 227px width
        scaled_x = 227.0 - scaled_x
      
      # Return scaled coordinates as a tensor
      label = torch.tensor([scaled_x, scaled_y], dtype=torch.float32)
      
      return image, label