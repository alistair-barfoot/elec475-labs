from torchvision.datasets import VOCSegmentation
import os
import matplotlib.pyplot as plt
import PIL
import numpy as np
from torch.utils.data import Dataset

def main():
    try:
        # First try to use existing data without downloading
        dataset = VOCSegmentation(root='./data', year='2012', image_set='train', download=False)
        print(f'Total number of images in the dataset: {len(dataset)}')
        img, target = dataset[0]
        print(f'Image size: {img.size}, Target size: {target.size}')
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Dataset not found locally: {e}")
        print("Attempting to download the dataset...")
        try:
            # Try to download if local data doesn't exist
            dataset = VOCSegmentation(root='./data', year='2012', image_set='train', download=True)
            print(f'Total number of images in the dataset: {len(dataset)}')
            img, target = dataset[0]
            print(f'Image size: {img.size}, Target size: {target.size}')
        except Exception as download_error:
            print(f"Failed to download dataset: {download_error}")
            print("\nPlease check your internet connection or manually download the VOC2012 dataset.")
            print("You can download it from: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/")
            print("Extract it to: ./data/VOCdevkit/VOC2012/")
            return

if __name__ == "__main__":
    main()