import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCSegmentation
import numpy as np
from PIL import Image

import torchvision.transforms as transforms

class VOC2012Dataset(Dataset):
  def __init__(self, root, year='2012', image_set='train', download=False, transform=None, target_transform=None):
    self.voc_dataset = VOCSegmentation(
      root=root,
      year=year,
      image_set=image_set,
      download=download,
      transform=None,
      target_transform=None
    )
    self.transform = transform
    self.target_transform = target_transform
    
  def __len__(self):
    return len(self.voc_dataset)
  
  def __getitem__(self, idx):
    image, target = self.voc_dataset[idx]
    
    if self.transform:
      image = self.transform(image)
    
    if self.target_transform:
      target = self.target_transform(target)
    
    return image, target

def get_voc2012_dataloader(root='./data', 
              image_set='train', 
              batch_size=32, 
              shuffle=True, 
              num_workers=4,
              download=False):
  """
  Create a DataLoader for VOC2012 dataset
  
  Args:
    root: Root directory for dataset
    image_set: 'train', 'val', or 'trainval'
    batch_size: Batch size for DataLoader
    shuffle: Whether to shuffle the data
    num_workers: Number of worker processes
    download: Whether to download the dataset if not found
  
  Returns:
    DataLoader object
  """
  
  # Define transforms
  transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
               std=[0.229, 0.224, 0.225])
  ])
  
  target_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.NEAREST),
    transforms.ToTensor()
  ])
  
  # Create dataset
  dataset = VOC2012Dataset(
    root=root,
    image_set=image_set,
    download=download,
    transform=transform,
    target_transform=target_transform
  )
  
  # Create dataloader
  dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    pin_memory=True
  )
  
  return dataloader

# Example usage
if __name__ == "__main__":
  # Create train and validation dataloaders
  train_loader = get_voc2012_dataloader(
    root='./data',
    image_set='train',
    batch_size=16,
    shuffle=True,
    download=True
  )
  
  val_loader = get_voc2012_dataloader(
    root='./data',
    image_set='val',
    batch_size=16,
    shuffle=False,
    download=True
  )
  
  print(f"Train dataset size: {len(train_loader.dataset)}")
  print(f"Validation dataset size: {len(val_loader.dataset)}")
  
  # Test loading a batch
  for images, targets in train_loader:
    print(f"Batch - Images shape: {images.shape}, Targets shape: {targets.shape}")
    break