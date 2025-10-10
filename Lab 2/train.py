# dataset.py
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

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
                image = Image.open(img_path)
                # Convert PIL to tensor (same result as decode_image)
                to_tensor = transforms.ToTensor()
                image = to_tensor(image)
                
                label = self.img_labels.iloc[idx, 1]
                if self.transform:
                    image = self.transform(image)
                if self.target_transform:
                    label = self.target_transform(label)
                return image, label