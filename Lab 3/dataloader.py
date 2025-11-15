"""
VOC 2012 semantic segmentation dataloader for MBV3SmallSeg model.
No augmentation - clean preprocessing only.

Features:
- Loads from local archive/VOC2012_train_val directory structure
- Resizes images and masks to fixed size
- Applies ImageNet normalization for pretrained MobileNetV3
- Handles VOC mask encoding (0-20 classes + 255 for ignore/boundary)
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import numpy as np


class VOC2012SegDataset(Dataset):
    """
    VOC 2012 Segmentation Dataset from local archive folder.
    
    Args:
        root: path to archive folder (default './archive')
        image_set: 'train' or 'val'
        img_size: tuple (H, W) for resizing (default 256x256)
    """
    def __init__(self, 
                 root='./archive', 
                 image_set='train', 
                 img_size=(256, 256)):
        super().__init__()
        
        # Paths to your local VOC structure
        self.root = root
        self.img_size = img_size
        self.image_set = image_set
        
        # Base path for train/val data
        if image_set == 'train' or image_set == 'val':
            voc_root = os.path.join(root, 'VOC2012_train_val', 'VOC2012_train_val')
        elif image_set == 'test':
            voc_root = os.path.join(root, 'VOC2012_test', 'VOC2012_test')
        else:
            raise ValueError("image_set must be 'train', 'val', or 'test'")
        
        # Read image IDs from ImageSets
        split_file = os.path.join(voc_root, 'ImageSets', 'Segmentation', f'{image_set}.txt')
        
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]
        
        # Directories
        self.image_dir = os.path.join(voc_root, 'JPEGImages')
        self.mask_dir = os.path.join(voc_root, 'SegmentationClass')
        
        # ImageNet normalization (required for pretrained MobileNetV3)
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        print(f"Loaded {len(self.image_ids)} images for {image_set} set")
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        
        # Load image and mask
        img_path = os.path.join(self.image_dir, f'{img_id}.jpg')
        mask_path = os.path.join(self.mask_dir, f'{img_id}.png')
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        
        # Resize image with bilinear interpolation
        image = TF.resize(image, self.img_size, interpolation=Image.BILINEAR)
        
        # Resize mask with NEAREST to preserve class labels
        mask = TF.resize(mask, self.img_size, interpolation=Image.NEAREST)
        
        # Convert to tensors
        image = TF.to_tensor(image)  # (3, H, W) in [0, 1]
        image = self.normalize(image)
        
        # Mask to long tensor (VOC uses 0-20 for classes, 255 for ignore)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))  # (H, W)
        
        return image, mask


def get_voc_dataloader(root='./archive',
                       image_set='train',
                       img_size=(256, 256),
                       batch_size=16,
                       shuffle=None,
                       num_workers=4,
                       pin_memory=True):
    """
    Create a DataLoader for VOC 2012 segmentation.
    
    Args:
        root: path to archive folder (default './archive')
        image_set: 'train' or 'val'
        img_size: (H, W) tuple for resizing (default 256x256)
        batch_size: batch size
        shuffle: whether to shuffle (default: True for train, False for val)
        num_workers: number of DataLoader workers
        pin_memory: use pinned memory for GPU
    
    Returns:
        DataLoader instance
    """
    if shuffle is None:
        shuffle = (image_set == 'train')
    
    dataset = VOC2012SegDataset(
        root=root,
        image_set=image_set,
        img_size=img_size
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(image_set == 'train')
    )
    
    return dataloader


# VOC class names for reference
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
    'sofa', 'train', 'tvmonitor'
]


# ---------------------------
# Quick test
# ---------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    print("Creating VOC 2012 dataloaders...")
    
    # Create train and val loaders
    train_loader = get_voc_dataloader(
        root='./archive',
        image_set='train',
        img_size=(256, 256),
        batch_size=8,
        shuffle=True,
        num_workers=0  # set to 0 for debugging, increase for training
    )
    
    val_loader = get_voc_dataloader(
        root='./archive',
        image_set='val',
        img_size=(256, 256),
        batch_size=8,
        shuffle=False,
        num_workers=0
    )
    
    print(f"\nTrain dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size: {len(val_loader.dataset)}")
    print(f"Batches per epoch (train): {len(train_loader)}")
    print(f"Batches per epoch (val): {len(val_loader)}")
    
    # Test loading a batch
    print("\nLoading sample batch...")
    images, masks = next(iter(train_loader))
    print(f"Images batch shape: {images.shape}")  # (B, 3, H, W)
    print(f"Masks batch shape: {masks.shape}")    # (B, H, W)
    print(f"Image value range: [{images.min():.2f}, {images.max():.2f}]")
    print(f"Unique mask values: {torch.unique(masks).tolist()}")
    
    # Visualize one sample
    print("\nVisualizing first sample...")
    img = images[0]
    mask = masks[0]
    
    # Denormalize image for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_denorm = img * std + mean
    img_denorm = torch.clamp(img_denorm, 0, 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(img_denorm.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title('Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask.cpu().numpy(), cmap='tab20', vmin=0, vmax=20)
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    axes[2].imshow(img_denorm.permute(1, 2, 0).cpu().numpy())
    axes[2].imshow(mask.cpu().numpy(), cmap='tab20', alpha=0.5, vmin=0, vmax=20)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('voc_dataloader_sample.png', dpi=100)
    print("Sample saved as 'voc_dataloader_sample.png'")