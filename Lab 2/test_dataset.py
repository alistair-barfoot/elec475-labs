import os
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import Dataset
from train import CustomDataset

# Test the dataset with just the first few entries
def test_dataset():
    try:
        # Create dataset
        dataset = CustomDataset(annotations_file='train_noses.txt', img_dir='images', transform=None)
        print(f"Dataset length: {len(dataset)}")
        
        # Try to get the first few items
        for i in range(min(3, len(dataset))):
            print(f"\nTrying to load item {i}...")
            image, label = dataset[i]
            print(f"Successfully loaded item {i}: image shape = {image.shape}, label = {label}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset()