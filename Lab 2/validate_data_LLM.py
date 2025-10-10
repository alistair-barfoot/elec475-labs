#!/usr/bin/env python3
"""
Data validation script for the nose detection dataset.
This script provides a comprehensive check of the dataset and visualizes samples.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import ast
import argparse

class NoseDataset(Dataset):
    """Custom dataset class for loading images and nose coordinate labels"""
    def __init__(self, images_dir, labels_file, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.data = []
        self.failed_files = []
        
        # Parse the labels file
        with open(labels_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    # Parse format: filename,"(x, y)"
                    parts = line.split(',', 1)
                    if len(parts) == 2:
                        filename = parts[0].strip()
                        coord_str = parts[1].strip().strip('"')
                        # Parse coordinates from string "(x, y)"
                        try:
                            coords = ast.literal_eval(coord_str)
                            if isinstance(coords, tuple) and len(coords) == 2:
                                x, y = coords
                                self.data.append((filename, float(x), float(y)))
                            else:
                                self.failed_files.append((line_num, filename, "Invalid coordinate format"))
                        except (ValueError, SyntaxError) as e:
                            self.failed_files.append((line_num, filename, f"Parse error: {e}"))
                    else:
                        self.failed_files.append((line_num, line, "Invalid line format"))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        filename, x, y = self.data[idx]
        
        # Load image
        image_path = os.path.join(self.images_dir, filename)
        try:
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (227, 227), (0, 0, 0))
            original_size = (227, 227)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Convert coordinates to tensor
        coordinates = torch.tensor([x, y], dtype=torch.float32)
        
        return image, coordinates, filename, original_size

def validate_dataset_comprehensive(dataset, images_dir):
    """Comprehensive dataset validation"""
    print(f"=== Dataset Validation Report ===")
    print(f"Total samples: {len(dataset)}")
    print(f"Failed to parse: {len(dataset.failed_files)}")
    
    if dataset.failed_files:
        print("\nFailed files:")
        for line_num, filename, error in dataset.failed_files[:10]:  # Show first 10
            print(f"  Line {line_num}: {filename} - {error}")
        if len(dataset.failed_files) > 10:
            print(f"  ... and {len(dataset.failed_files) - 10} more")
    
    # Check image availability
    missing_images = []
    coordinate_stats = {'x': [], 'y': []}
    image_sizes = []
    
    print(f"\nChecking image files and coordinates...")
    for i in range(min(len(dataset), 1000)):  # Check first 1000 or all if less
        filename, x, y = dataset.data[i]
        image_path = os.path.join(images_dir, filename)
        
        if not os.path.exists(image_path):
            missing_images.append(filename)
        else:
            try:
                with Image.open(image_path) as img:
                    image_sizes.append(img.size)
            except Exception as e:
                missing_images.append(f"{filename} (corrupt: {e})")
        
        coordinate_stats['x'].append(x)
        coordinate_stats['y'].append(y)
    
    print(f"Missing/corrupt images: {len(missing_images)}")
    if missing_images:
        print("First 10 missing files:")
        for img in missing_images[:10]:
            print(f"  {img}")
    
    # Coordinate statistics
    x_coords = np.array(coordinate_stats['x'])
    y_coords = np.array(coordinate_stats['y'])
    
    print(f"\nCoordinate Statistics:")
    print(f"X coordinates - Min: {x_coords.min():.1f}, Max: {x_coords.max():.1f}, Mean: {x_coords.mean():.1f}")
    print(f"Y coordinates - Min: {y_coords.min():.1f}, Max: {y_coords.max():.1f}, Mean: {y_coords.mean():.1f}")
    print(f"Negative X coords: {(x_coords < 0).sum()}")
    print(f"Negative Y coords: {(y_coords < 0).sum()}")
    
    # Image size statistics
    if image_sizes:
        widths = [size[0] for size in image_sizes]
        heights = [size[1] for size in image_sizes]
        print(f"\nImage Size Statistics:")
        print(f"Widths - Min: {min(widths)}, Max: {max(widths)}, Mean: {np.mean(widths):.1f}")
        print(f"Heights - Min: {min(heights)}, Max: {max(heights)}, Mean: {np.mean(heights):.1f}")
        print(f"Unique image sizes: {len(set(image_sizes))}")
    
    return len(missing_images) == 0 and len(dataset.failed_files) == 0

def visualize_samples_detailed(dataset, num_samples=8, save_path='dataset_validation.png'):
    """Visualize sample images with detailed information"""
    if len(dataset) < num_samples:
        num_samples = len(dataset)
    
    fig, axes = plt.subplots(2, num_samples//2, figsize=(20, 8))
    axes = axes.flatten()
    
    # Create transform for visualization (no normalization)
    viz_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor()
    ])
    
    sample_indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(sample_indices):
        filename, x, y = dataset.data[idx]
        
        # Load image with visualization transform
        image_path = os.path.join(dataset.images_dir, filename)
        try:
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
        except:
            image = Image.new('RGB', (227, 227), (0, 0, 0))
            original_size = (227, 227)
        
        image_tensor = viz_transform(image)
        image_np = image_tensor.permute(1, 2, 0).numpy()
        
        axes[i].imshow(image_np)
        
        # Scale coordinates to displayed image size (227x227)
        scale_x = 227 / original_size[0]
        scale_y = 227 / original_size[1]
        x_scaled = x * scale_x
        y_scaled = y * scale_y
        
        # Plot nose location
        axes[i].plot(x_scaled, y_scaled, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=2)
        
        # Add title with information
        title = f'{filename}\nOrig: {original_size[0]}x{original_size[1]}\nNose: ({x:.0f}, {y:.0f})'
        axes[i].set_title(title, fontsize=8)
        axes[i].axis('off')
        
        # Add coordinate validation
        if x_scaled < 0 or y_scaled < 0 or x_scaled >= 227 or y_scaled >= 227:
            axes[i].add_patch(plt.Rectangle((0, 0), 227, 227, fill=False, edgecolor='red', linewidth=3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Detailed sample visualizations saved to {save_path}')

def plot_coordinate_distribution(dataset, save_path='coordinate_distribution.png'):
    """Plot distribution of nose coordinates"""
    x_coords = [data[1] for data in dataset.data]
    y_coords = [data[2] for data in dataset.data]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # X coordinate distribution
    ax1.hist(x_coords, bins=50, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Frequency')
    ax1.set_title('X Coordinate Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Y coordinate distribution
    ax2.hist(y_coords, bins=50, alpha=0.7, edgecolor='black', color='orange')
    ax2.set_xlabel('Y Coordinate')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Y Coordinate Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 2D scatter plot
    ax3.scatter(x_coords, y_coords, alpha=0.5, s=1)
    ax3.set_xlabel('X Coordinate')
    ax3.set_ylabel('Y Coordinate')
    ax3.set_title('Nose Position Distribution')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Coordinate distribution plots saved to {save_path}')

def main():
    parser = argparse.ArgumentParser(description='Validate nose detection dataset')
    parser.add_argument('--images_dir', type=str, default='images',
                       help='Path to the images directory')
    parser.add_argument('--train_labels', type=str, default='train_noses.txt',
                       help='Path to training labels file')
    parser.add_argument('--test_labels', type=str, default='test_noses.txt',
                       help='Path to test labels file')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.images_dir):
        print(f"Error: Images directory not found: {args.images_dir}")
        return
    
    if not os.path.exists(args.train_labels):
        print(f"Error: Training labels file not found: {args.train_labels}")
        return
    
    if not os.path.exists(args.test_labels):
        print(f"Error: Test labels file not found: {args.test_labels}")
        return
    
    print("Loading datasets...")
    
    # Load training dataset
    train_dataset = NoseDataset(args.images_dir, args.train_labels)
    print("\n" + "="*50)
    print("TRAINING DATASET")
    train_valid = validate_dataset_comprehensive(train_dataset, args.images_dir)
    
    # Load test dataset
    test_dataset = NoseDataset(args.images_dir, args.test_labels)
    print("\n" + "="*50)
    print("TEST DATASET")
    test_valid = validate_dataset_comprehensive(test_dataset, args.images_dir)
    
    if train_valid and test_valid:
        print("\n✅ All datasets passed validation!")
        
        # Visualize samples
        print("\nGenerating visualizations...")
        visualize_samples_detailed(train_dataset, num_samples=8, save_path='train_samples.png')
        visualize_samples_detailed(test_dataset, num_samples=8, save_path='test_samples.png')
        
        # Plot coordinate distributions
        plot_coordinate_distribution(train_dataset, save_path='train_coordinate_distribution.png')
        plot_coordinate_distribution(test_dataset, save_path='test_coordinate_distribution.png')
        
        print("\n🎉 Dataset validation complete! Ready for training.")
    else:
        print("\n❌ Dataset validation failed. Please fix the issues above before training.")

if __name__ == '__main__':
    main()