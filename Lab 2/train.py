# dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class PetNoseDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(
            annotations_file, 
            header=None, 
            names=['filename', 'coordinates']
        )
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        row = self.img_labels.iloc[idx]
        filename = row['filename'].strip()
        coords = row['coordinates'].replace("(", "").replace(")", "").split(",")
        x, y = map(float, [c.strip() for c in coords])

        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert("RGB")

        w, h = image.size
        target = torch.tensor([x / w, y / h], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, target
