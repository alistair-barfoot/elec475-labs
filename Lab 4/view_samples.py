"""
Script to visualize random image-caption pairs from COCO dataset

Usage:
    python view_samples.py --dataset val --num-samples 6
    python view_samples.py --dataset train --num-samples 9 --save-fig samples.png
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import time

from dataloader import COCODataset, collate_fn, get_default_transforms
from train import get_caption_from_labels


def visualize_samples(dataset_name='val', num_samples=6, save_path=None):
    """
    Load and visualize random image-caption pairs from COCO dataset
    
    Args:
        dataset_name: 'train' or 'val'
        num_samples: Number of samples to display
        save_path: Optional path to save figure
    """
    
    print(f"Loading {dataset_name} dataset...")
    
    # Create dataset
    transform = get_default_transforms(image_size=(224, 224))
    dataset = COCODataset(
        dataset=dataset_name,
        transform=transform,
        load_annotations=True,
        image_size=(224, 224)
    )
    
    print(f"Dataset loaded: {len(dataset)} images")
    
    # Select random samples
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    # Calculate grid dimensions
    n_cols = 3
    n_rows = (num_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Random Image-Caption Pairs from {dataset_name.upper()} Dataset', 
                 fontsize=16, fontweight='bold')
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(indices):
            sample_idx = indices[idx]
            
            # Get sample
            sample = dataset[sample_idx]
            
            # Denormalize image for display
            image = sample['image']
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
            image = image * std + mean
            image = torch.clamp(image, 0, 1)
            
            # Convert to numpy for display
            image_np = image.permute(1, 2, 0).numpy()
            
            # Get caption from labels
            labels = sample.get('labels', [])
            caption = get_caption_from_labels(labels, dataset)
            
            # Get additional info
            filename = sample['filename']
            num_objects = sample.get('num_objects', 0)
            
            # Display image
            ax.imshow(image_np)
            
            # Create title with wrapped caption
            title = f"Sample #{sample_idx}\n{filename}\n"
            title += f"Objects: {num_objects}\n"
            title += f"Caption: {caption}"
            
            # Wrap long captions
            if len(caption) > 40:
                words = caption.split()
                lines = []
                current_line = []
                current_length = 0
                
                for word in words:
                    if current_length + len(word) + 1 > 40:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                        current_length = len(word)
                    else:
                        current_line.append(word)
                        current_length += len(word) + 1
                
                if current_line:
                    lines.append(' '.join(current_line))
                
                title = f"Sample #{sample_idx}\n{filename}\nObjects: {num_objects}\n" + '\n'.join(lines)
            
            ax.set_title(title, fontsize=9, pad=10)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()
    
    # Print detailed info for samples
    print(f"\n{'='*80}")
    print("DETAILED SAMPLE INFORMATION")
    print(f"{'='*80}")
    
    for i, sample_idx in enumerate(indices[:5]):  # Show details for first 5
        sample = dataset[sample_idx]
        labels = sample.get('labels', [])
        bboxes = sample.get('bboxes', [])
        
        print(f"\nSample {i+1} (Index {sample_idx}):")
        print(f"  Filename: {sample['filename']}")
        print(f"  Original Size: {sample['original_size']}")
        print(f"  Number of Objects: {sample.get('num_objects', 0)}")
        
        if labels:
            print(f"  Category IDs: {labels}")
            category_names = [dataset.get_category_name(label) for label in labels]
            print(f"  Category Names: {category_names}")
            print(f"  Caption: {get_caption_from_labels(labels, dataset)}")
        
        if bboxes and len(bboxes) > 0:
            print(f"  First 3 Bounding Boxes: {bboxes[:3]}")


def compare_train_val_samples(num_samples_per_dataset=3, save_path=None):
    """
    Compare samples from training and validation datasets side by side
    
    Args:
        num_samples_per_dataset: Number of samples to show from each dataset
        save_path: Optional path to save figure
    """
    
    print("Loading train and val datasets...")
    
    transform = get_default_transforms(image_size=(224, 224))
    
    train_dataset = COCODataset(
        dataset='train',
        transform=transform,
        load_annotations=True,
        image_size=(224, 224)
    )
    
    val_dataset = COCODataset(
        dataset='val',
        transform=transform,
        load_annotations=True,
        image_size=(224, 224)
    )
    
    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Val dataset: {len(val_dataset)} images")
    
    # Select random samples
    train_indices = random.sample(range(len(train_dataset)), 
                                  min(num_samples_per_dataset, len(train_dataset)))
    val_indices = random.sample(range(len(val_dataset)), 
                               min(num_samples_per_dataset, len(val_dataset)))
    
    # Create figure
    fig, axes = plt.subplots(2, num_samples_per_dataset, 
                            figsize=(5 * num_samples_per_dataset, 10))
    
    if num_samples_per_dataset == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle('Training vs Validation Dataset Samples', fontsize=16, fontweight='bold')
    
    # Plot training samples
    for col, sample_idx in enumerate(train_indices):
        sample = train_dataset[sample_idx]
        
        # Denormalize
        image = sample['image']
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        image = image * std + mean
        image = torch.clamp(image, 0, 1)
        image_np = image.permute(1, 2, 0).numpy()
        
        labels = sample.get('labels', [])
        caption = get_caption_from_labels(labels, train_dataset)
        
        axes[0, col].imshow(image_np)
        axes[0, col].set_title(f"TRAIN #{sample_idx}\n{caption[:50]}...", 
                              fontsize=9, color='blue', fontweight='bold')
        axes[0, col].axis('off')
    
    # Plot validation samples
    for col, sample_idx in enumerate(val_indices):
        sample = val_dataset[sample_idx]
        
        # Denormalize
        image = sample['image']
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        image = image * std + mean
        image = torch.clamp(image, 0, 1)
        image_np = image.permute(1, 2, 0).numpy()
        
        labels = sample.get('labels', [])
        caption = get_caption_from_labels(labels, val_dataset)
        
        axes[1, col].imshow(image_np)
        axes[1, col].set_title(f"VAL #{sample_idx}\n{caption[:50]}...", 
                              fontsize=9, color='green', fontweight='bold')
        axes[1, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to: {save_path}")
    
    plt.show()


def show_class_distribution(dataset_name='val', top_k=20):
    """
    Show class distribution in the dataset
    
    Args:
        dataset_name: 'train', 'val', or 'test'
        top_k: Number of top classes to display
    """
    
    print(f"Loading {dataset_name} dataset and computing class distribution...")
    
    transform = get_default_transforms(image_size=(224, 224))
    dataset = COCODataset(
        dataset=dataset_name,
        transform=transform,
        load_annotations=True,
        image_size=(224, 224)
    )
    
    class_dist = dataset.get_class_distribution()
    
    print(f"\n{'='*80}")
    print(f"CLASS DISTRIBUTION - {dataset_name.upper()} DATASET")
    print(f"{'='*80}")
    print(f"Total classes: {len(class_dist)}")
    print(f"Total instances: {sum(class_dist.values())}")
    print(f"\nTop {top_k} classes:")
    
    for i, (class_name, count) in enumerate(list(class_dist.items())[:top_k], 1):
        percentage = (count / sum(class_dist.values())) * 100
        print(f"{i:3d}. {class_name:20s}: {count:6d} instances ({percentage:5.2f}%)")
    
    # Plot distribution
    top_classes = list(class_dist.items())[:top_k]
    class_names = [c[0] for c in top_classes]
    counts = [c[1] for c in top_classes]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(class_names, counts, color='steelblue')
    ax.set_xlabel('Number of Instances', fontsize=12)
    ax.set_title(f'Top {top_k} Classes in {dataset_name.upper()} Dataset', 
                fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add count labels
    for bar, count in zip(bars, counts):
        width = bar.get_width()
        ax.text(width + max(counts) * 0.01, bar.get_y() + bar.get_height()/2, 
               f'{count}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize COCO dataset image-caption pairs"
    )
    
    parser.add_argument('--dataset', type=str, default='val', 
                       choices=['train', 'val', 'test'],
                       help='Dataset to visualize (train, val, or test)')
    parser.add_argument('--num-samples', type=int, default=6,
                       help='Number of samples to display')
    parser.add_argument('--save-fig', type=str, default=None,
                       help='Path to save figure (optional)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare train and val datasets side by side')
    parser.add_argument('--class-dist', action='store_true',
                       help='Show class distribution')
    parser.add_argument('--top-k', type=int, default=20,
                       help='Number of top classes to show in distribution')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    random.seed(time.time())
    
    if args.class_dist:
        show_class_distribution(args.dataset, args.top_k)
    elif args.compare:
        compare_train_val_samples(
            num_samples_per_dataset=args.num_samples,
            save_path=args.save_fig
        )
    else:
        visualize_samples(
            dataset_name=args.dataset,
            num_samples=args.num_samples,
            save_path=args.save_fig
        )


if __name__ == "__main__":
    main()
