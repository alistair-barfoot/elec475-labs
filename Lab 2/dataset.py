import os
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import Dataset
from model import snoutNet
import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision import tv_tensors

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
      
      # Parse coordinates from string format "(x, y)"
      label_str = self.img_labels.iloc[idx, 1]
      coords = label_str.strip('()').split(', ')
      original_x = float(coords[0])
      original_y = float(coords[1])

      # Convert to tensor and create proper tv_tensors format
      image = v2.Compose([
          v2.ToImage(),
          v2.ToDtype(torch.float32, scale=True)
      ])(image)
      
      # Create bounding box using tv_tensors (this will be transformed correctly)
      # Format: [x1, y1, x2, y2] - using a small box around the nose point
      boxes = tv_tensors.BoundingBoxes(
          [[original_x-0.5, original_y-0.5, original_x+0.5, original_y+0.5]], 
          format="XYXY", 
          canvas_size=(original_height, original_width)
      )
      
      # Apply v2 transforms (now they will handle both image and boxes correctly)
      if self.transform:
          image, boxes = self.transform(image, boxes)
      else:
          # Default transform using v2
          default_transform = v2.Compose([
              v2.Resize((227, 227)),
          ])
          image, boxes = default_transform(image, boxes)
      
      # Extract nose coordinates from transformed boxes (center of the box)
      box = boxes[0]
      nose_x = (box[0] + box[2]) / 2  # center x
      nose_y = (box[1] + box[3]) / 2  # center y
      
      # Return as tensor
      label = torch.tensor([nose_x.item(), nose_y.item()], dtype=torch.float32)
      
      return image, label