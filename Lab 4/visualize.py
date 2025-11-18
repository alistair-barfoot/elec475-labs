import os
import json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
from pathlib import Path

class COCOVisualizer:
    def __init__(self, archive_path_file="path_to_archive.txt"):
        """Initialize the COCO visualizer with the archive path."""
        with open(archive_path_file, "r") as f:
            self.archive_path = f.read().strip()
        
        print(f"Archive path: {self.archive_path}")
        
        # Define paths
        self.coco_root = os.path.join(self.archive_path, "coco2014")
        self.train_images_path = os.path.join(self.coco_root, "images", "train2014")
        self.val_images_path = os.path.join(self.coco_root, "images", "val2014")
        self.test_images_path = os.path.join(self.coco_root, "images", "test2014")
        
        # Annotation paths
        self.train_annotations_path = os.path.join(self.coco_root, "annotations", "instances_train2014.json")
        self.val_annotations_path = os.path.join(self.coco_root, "annotations", "instances_val2014.json")
        
        # Load annotations if available
        self.train_annotations = None
        self.val_annotations = None
        self._load_annotations()
        
    def _load_annotations(self):
        """Load COCO annotations if they exist."""
        try:
            if os.path.exists(self.train_annotations_path):
                print("Loading training annotations...")
                with open(self.train_annotations_path, 'r') as f:
                    self.train_annotations = json.load(f)
                print(f"Loaded {len(self.train_annotations['images'])} training images")
                
            if os.path.exists(self.val_annotations_path):
                print("Loading validation annotations...")
                with open(self.val_annotations_path, 'r') as f:
                    self.val_annotations = json.load(f)
                print(f"Loaded {len(self.val_annotations['images'])} validation images")
                
        except Exception as e:
            print(f"Could not load annotations: {e}")
    
    def get_image_list(self, dataset='val', limit=None):
        """Get list of image files from specified dataset."""
        if dataset == 'train':
            image_dir = self.train_images_path
        elif dataset == 'val':
            image_dir = self.val_images_path
        elif dataset == 'test':
            image_dir = self.test_images_path
        else:
            raise ValueError("Dataset must be 'train', 'val', or 'test'")
        
        if not os.path.exists(image_dir):
            print(f"Directory {image_dir} does not exist")
            return []
            
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if limit:
            image_files = image_files[:limit]
            
        return image_files
    
    def load_image(self, image_filename, dataset='val'):
        """Load a single image from the dataset."""
        if dataset == 'train':
            image_path = os.path.join(self.train_images_path, image_filename)
        elif dataset == 'val':
            image_path = os.path.join(self.val_images_path, image_filename)
        elif dataset == 'test':
            image_path = os.path.join(self.test_images_path, image_filename)
        else:
            raise ValueError("Dataset must be 'train', 'val', or 'test'")
        
        if not os.path.exists(image_path):
            print(f"Image {image_path} does not exist")
            return None
            
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def get_image_annotations(self, image_filename, dataset='val'):
        """Get annotations for a specific image."""
        annotations_data = None
        if dataset == 'train' and self.train_annotations:
            annotations_data = self.train_annotations
        elif dataset == 'val' and self.val_annotations:
            annotations_data = self.val_annotations
        
        if not annotations_data:
            return [], {}
            
        # Find image ID
        image_id = None
        for img_info in annotations_data['images']:
            if img_info['file_name'] == image_filename:
                image_id = img_info['id']
                break
        
        if image_id is None:
            return [], {}
        
        # Get annotations for this image
        image_annotations = []
        for ann in annotations_data['annotations']:
            if ann['image_id'] == image_id:
                image_annotations.append(ann)
        
        # Create category lookup
        categories = {cat['id']: cat['name'] for cat in annotations_data['categories']}
        
        return image_annotations, categories
    
    def visualize_single_image(self, image_filename=None, dataset='val', show_annotations=True):
        """Visualize a single image with optional annotations."""
        # Get random image if none specified
        if image_filename is None:
            image_list = self.get_image_list(dataset, limit=100)
            if not image_list:
                print(f"No images found in {dataset} dataset")
                return
            image_filename = random.choice(image_list)
        
        # Load image
        image = self.load_image(image_filename, dataset)
        if image is None:
            print(f"Could not load image: {image_filename}")
            return
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        ax.set_title(f"{dataset.capitalize()} Dataset: {image_filename}")
        ax.axis('off')
        
        # Add annotations if available and requested
        if show_annotations:
            annotations, categories = self.get_image_annotations(image_filename, dataset)
            
            if annotations:
                colors = plt.cm.tab10(np.linspace(0, 1, len(set(ann['category_id'] for ann in annotations))))
                color_map = {}
                
                for i, ann in enumerate(annotations):
                    category_id = ann['category_id']
                    category_name = categories.get(category_id, f"Category {category_id}")
                    
                    if category_id not in color_map:
                        color_map[category_id] = colors[len(color_map) % len(colors)]
                    
                    # Draw bounding box
                    bbox = ann['bbox']  # [x, y, width, height]
                    rect = patches.Rectangle(
                        (bbox[0], bbox[1]), bbox[2], bbox[3],
                        linewidth=2, edgecolor=color_map[category_id], 
                        facecolor='none', alpha=0.8
                    )
                    ax.add_patch(rect)
                    
                    # Add label
                    ax.text(
                        bbox[0], bbox[1] - 5, category_name,
                        fontsize=10, color=color_map[category_id],
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7)
                    )
        
        plt.tight_layout()
        plt.show()
        
        return image_filename
    
    def visualize_grid(self, dataset='val', num_images=9, show_annotations=False):
        """Visualize a grid of images from the dataset."""
        image_list = self.get_image_list(dataset, limit=num_images*2)  # Get more than needed for random selection
        if len(image_list) < num_images:
            num_images = len(image_list)
            
        if num_images == 0:
            print(f"No images found in {dataset} dataset")
            return
        
        selected_images = random.sample(image_list, num_images)
        
        # Calculate grid dimensions
        rows = int(np.ceil(np.sqrt(num_images)))
        cols = int(np.ceil(num_images / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        if num_images == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, image_filename in enumerate(selected_images):
            image = self.load_image(image_filename, dataset)
            if image is not None:
                axes[i].imshow(image)
                axes[i].set_title(image_filename, fontsize=8)
                axes[i].axis('off')
                
                # Add simple annotation count if available
                if show_annotations:
                    annotations, _ = self.get_image_annotations(image_filename, dataset)
                    if annotations:
                        axes[i].text(0.02, 0.98, f"{len(annotations)} objects", 
                                   transform=axes[i].transAxes, fontsize=8,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7),
                                   verticalalignment='top')
        
        # Hide unused subplots
        for i in range(num_images, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f"COCO 2014 {dataset.capitalize()} Dataset - Random Sample", fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def dataset_info(self):
        """Print information about the dataset."""
        print("\n=== COCO 2014 Dataset Information ===")
        
        # Check available directories
        for name, path in [("Train images", self.train_images_path), 
                          ("Validation images", self.val_images_path),
                          ("Test images", self.test_images_path)]:
            if os.path.exists(path):
                count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"{name}: {count} images")
            else:
                print(f"{name}: Directory not found")
        
        # Annotation info
        if self.train_annotations:
            print(f"Training annotations: {len(self.train_annotations['categories'])} categories, "
                  f"{len(self.train_annotations['annotations'])} annotations")
        
        if self.val_annotations:
            print(f"Validation annotations: {len(self.val_annotations['categories'])} categories, "
                  f"{len(self.val_annotations['annotations'])} annotations")

# Example usage
if __name__ == "__main__":
    # Initialize visualizer
    viz = COCOVisualizer()
    
    # Show dataset information
    viz.dataset_info()
    
    # Visualize a grid of random images from validation set
    print("\nShowing grid of validation images...")
    viz.visualize_grid(dataset='val', num_images=6, show_annotations=False)
    
    # Visualize a single image with annotations
    print("\nShowing single image with annotations...")
    viz.visualize_single_image(dataset='val', show_annotations=True)
    
    # You can also specify a particular image
    # viz.visualize_single_image("COCO_val2014_000000000042.jpg", dataset='val', show_annotations=True)

