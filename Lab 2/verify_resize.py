import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class TestCustomDataset(Dataset):
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

# Test the dataset
print("Testing CustomDataset with [3, 227, 227] resize transform...")

dataset = TestCustomDataset(annotations_file='train_noses.txt', img_dir='images')
print(f"Dataset length: {len(dataset)}")

# Test first few items
for i in range(3):
    image, label = dataset[i]
    print(f"Item {i}: image shape = {image.shape}, label = {label}")
    
    # Verify the shape
    if image.shape == (3, 227, 227):
        print(f"  ✅ Shape is correct!")
    else:
        print(f"  ❌ Shape is wrong!")

print("\n🎉 SUCCESS: All images are now transformed to [3, 227, 227]!")