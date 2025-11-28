import os
import json
import random
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import cast

class COCODataset(Dataset):
    """COCO 2014 Dataset for PyTorch DataLoader"""
    
    def __init__(self, 
                 archive_path_file: str = "path_to_archive.txt",
                 dataset: str = 'val',
                 transform: Optional[transforms.Compose] = None,
                 load_annotations: bool = True,
                 image_size: Tuple[int, int] = (224, 224)):
        """
        Initialize COCO Dataset
        
        Args:
            archive_path_file: Path to file containing archive root path
            dataset: 'train', 'val'
            transform: Optional torchvision transforms
            load_annotations: Whether to load and return annotations
            image_size: Target image size (height, width)
        """
        
        # Read archive path
        with open(archive_path_file, "r") as f:
            self.archive_path = f.read().strip()
        
        self.dataset = dataset
        self.transform = transform
        self.load_annotations = load_annotations
        self.image_size = image_size
        
        # Define paths
        self.coco_root = os.path.join(self.archive_path, "coco2014")
        self.images_path = os.path.join(self.coco_root, "images", f"{dataset}2014")
        
        # Initialize image list
        self.image_files = self._get_image_files()
        
        # Load annotations if requested
        self.annotations_data = None
        self.image_id_to_annotations = {}
        self.categories = {}
        
        if load_annotations and dataset in ['train', 'val']:
            self._load_annotations()
    
    def _get_image_files(self) -> List[str]:
        """Get list of image files in the dataset directory"""
        if not os.path.exists(self.images_path):
            print(f"Warning: Image directory {self.images_path} does not exist")
            return []
        
        image_files = [f for f in os.listdir(self.images_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Found {len(image_files)} images in {self.dataset} dataset")
        return sorted(image_files)
    
    def _load_annotations(self):
        """Load COCO annotations"""
        annotations_path = os.path.join(self.coco_root, "annotations", 
                                       f"instances_{self.dataset}2014.json")
        
        if not os.path.exists(annotations_path):
            print(f"Warning: Annotations file {annotations_path} does not exist")
            return
        
        try:
            print(f"Loading {self.dataset} annotations...")
            with open(annotations_path, 'r') as f:
                self.annotations_data = json.load(f)
            
            # Create category lookup
            self.categories = {cat['id']: cat['name'] 
                             for cat in self.annotations_data['categories']}
            
            # Create image filename to ID mapping
            filename_to_id = {}
            for img_info in self.annotations_data['images']:
                filename_to_id[img_info['file_name']] = img_info['id']
            
            # Group annotations by image ID
            for ann in self.annotations_data['annotations']:
                image_id = ann['image_id']
                if image_id not in self.image_id_to_annotations:
                    self.image_id_to_annotations[image_id] = []
                self.image_id_to_annotations[image_id].append(ann)
            
            # Filter image files to only include those with annotations
            self.image_files = [img_file for img_file in self.image_files 
                              if img_file in filename_to_id and 
                              filename_to_id[img_file] in self.image_id_to_annotations]
            
            print(f"Loaded annotations for {len(self.image_files)} images")
            print(f"Dataset has {len(self.categories)} categories")
            
        except Exception as e:
            print(f"Error loading annotations: {e}")
            self.annotations_data = None
    
    def __len__(self) -> int:
        """Return the number of images in the dataset"""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single item from the dataset
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing:
                - 'image': PIL Image or transformed tensor
                - 'filename': Image filename
                - 'annotations': List of annotations (if load_annotations=True)
                - 'labels': List of category IDs (if load_annotations=True)
                - 'bboxes': List of bounding boxes (if load_annotations=True)
        """
        if idx >= len(self.image_files):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.image_files)}")
        
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.images_path, image_filename)
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
            original_size = image.size  # (width, height)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', self.image_size, color='black')
            original_size = self.image_size
        
        # Prepare return data
        data = {
            'image': image,
            'filename': image_filename,
            'original_size': original_size
        }
        
        # Add annotations if available
        if self.load_annotations and self.annotations_data:
            annotations, labels, bboxes = self._get_image_annotations(image_filename)
            data.update({
                'annotations': annotations,
                'labels': labels,
                'bboxes': bboxes,
                'num_objects': len(annotations)
            })
        
        # Apply transforms
        if self.transform:
            data['image'] = self.transform(data['image'])
        
        return data
    
    def _get_image_annotations(self, image_filename: str) -> Tuple[List[Dict], List[int], List[List[float]]]:
        """Get annotations for a specific image"""
        if not self.annotations_data:
            return [], [], []
        
        # Find image ID
        image_id = None
        for img_info in self.annotations_data['images']:
            if img_info['file_name'] == image_filename:
                image_id = img_info['id']
                break
        
        if image_id is None or image_id not in self.image_id_to_annotations:
            return [], [], []
        
        annotations = self.image_id_to_annotations[image_id]
        labels = [ann['category_id'] for ann in annotations]
        bboxes = [ann['bbox'] for ann in annotations]  # [x, y, width, height]
        
        return annotations, labels, bboxes
    
    def get_category_name(self, category_id: int) -> str:
        """Get category name from category ID"""
        return self.categories.get(category_id, f"Category {category_id}")
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes in the dataset"""
        if not self.load_annotations:
            return {}
        
        class_counts = {}
        for image_file in self.image_files:
            _, labels, _ = self._get_image_annotations(image_file)
            for label in labels:
                category_name = self.get_category_name(label)
                class_counts[category_name] = class_counts.get(category_name, 0) + 1
        
        return dict(sorted(class_counts.items(), key=lambda x: x[1], reverse=True))

def get_default_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Get default image transforms for COCO dataset"""
    transform_list = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ]
    
    return transforms.Compose(transform_list)

def get_augmentation_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Get augmentation transforms for training"""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def create_dataloader(dataset: str = 'val',
                     batch_size: int = 32,
                     shuffle: bool = True,
                     num_workers: int = 0,
                     image_size: Tuple[int, int] = (224, 224),
                     load_annotations: bool = True,
                     use_augmentation: bool = False) -> DataLoader:
    """
    Create a PyTorch DataLoader for COCO dataset
    
    Args:
        dataset: 'train', 'val', or 'test'
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes for data loading
        image_size: Target image size (height, width)
        load_annotations: Whether to load annotations
        use_augmentation: Whether to apply data augmentation
        
    Returns:
        PyTorch DataLoader
    """
    
    # Choose transforms
    if use_augmentation and dataset == 'train':
        transform = get_augmentation_transforms(image_size)
    else:
        transform = get_default_transforms(image_size)
    
    # Create dataset
    coco_dataset = COCODataset(
        dataset=dataset,
        transform=transform,
        load_annotations=load_annotations,
        image_size=image_size
    )
    
    # Create dataloader
    dataloader = DataLoader(
        coco_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn if load_annotations else None
    )
    
    return dataloader

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function to handle variable number of annotations per image
    """
    images = torch.stack([item['image'] for item in batch])
    filenames = [item['filename'] for item in batch]
    
    # Handle annotations
    all_annotations = [item.get('annotations', []) for item in batch]
    all_labels = [item.get('labels', []) for item in batch]
    all_bboxes = [item.get('bboxes', []) for item in batch]
    num_objects = [item.get('num_objects', 0) for item in batch]
    
    return {
        'images': images,
        'filenames': filenames,
        'annotations': all_annotations,
        'labels': all_labels,
        'bboxes': all_bboxes,
        'num_objects': num_objects
    }

# Example usage
if __name__ == "__main__":
    # Create validation dataloader
    val_loader = create_dataloader(
        dataset='val',
        batch_size=8,
        shuffle=False,
        num_workers=0,
        image_size=(224, 224),
        load_annotations=True
    )
    
    print(f"Dataset size: {len(cast(COCODataset, val_loader.dataset))} images")
    
    # Test loading a batch
    try:
        batch = next(iter(val_loader))
        print(f"\nBatch info:")
        print(f"Images shape: {batch['images'].shape}")
        print(f"Number of files: {len(batch['filenames'])}")
        print(f"Objects per image: {batch['num_objects']}")
        
        # Show first image info
        if len(batch['filenames']) > 0:
            print(f"\nFirst image: {batch['filenames'][0]}")
            if batch['labels'][0]:
                print(f"Labels: {batch['labels'][0][:5]}...")  # Show first 5 labels
                
    except Exception as e:
        print(f"Error loading batch: {e}")
    
    # Show class distribution
    dataset = val_loader.dataset
    # Cast to COCODataset to access get_class_distribution
    if isinstance(dataset, COCODataset):
        class_dist = dataset.get_class_distribution()
        print(f"\nTop 10 most common classes:")
        for i, (class_name, count) in enumerate(list(class_dist.items())[:10]):
            print(f"{i+1:2d}. {class_name}: {count} instances")