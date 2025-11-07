"""
Training script for MBV3SmallSeg on VOC 2012 semantic segmentation.

Features:
- Logs training/validation loss per epoch
- Computes mean IoU (mIoU) on validation set
- Saves best model checkpoint based on mIoU
- Supports GPU training with automatic device detection
- Includes early stopping and learning rate scheduling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
from pathlib import Path

from MobileNet_model import MBV3SmallSeg
from dataloader import get_voc_dataloader


# ---------------------------
# mIoU metric computation
# ---------------------------
class IoUMetric:
    """
    Compute mean Intersection over Union (mIoU) for semantic segmentation.
    """
    def __init__(self, num_classes=21, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def update(self, preds, targets):
        """
        Update confusion matrix with batch predictions.
        
        Args:
            preds: (B, H, W) tensor of predicted class indices
            targets: (B, H, W) tensor of ground truth class indices
        """
        preds = preds.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()
        
        # Mask out ignore index
        mask = (targets != self.ignore_index)
        preds = preds[mask]
        targets = targets[mask]
        
        # Update confusion matrix
        for pred, target in zip(preds, targets):
            if 0 <= pred < self.num_classes and 0 <= target < self.num_classes:
                self.confusion_matrix[target, pred] += 1
    
    def compute(self):
        """
        Compute per-class IoU and mean IoU.
        
        Returns:
            dict with 'miou' (mean IoU), 'iou_per_class' (numpy array)
        """
        # IoU = TP / (TP + FP + FN)
        # TP is diagonal, FP is column sum - TP, FN is row sum - TP
        intersection = np.diag(self.confusion_matrix)
        union = (self.confusion_matrix.sum(axis=1) + 
                 self.confusion_matrix.sum(axis=0) - 
                 intersection)
        
        # Avoid division by zero
        iou = np.zeros(self.num_classes, dtype=np.float32)
        valid = union > 0
        iou[valid] = intersection[valid] / union[valid]
        
        # Mean IoU over classes that appear in the dataset
        miou = iou[valid].mean() if valid.any() else 0.0
        
        return {
            'miou': miou,
            'iou_per_class': iou
        }


# ---------------------------
# Training and validation functions
# ---------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train for one epoch.
    
    Returns:
        avg_loss: average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    num_batches = len(dataloader)
    
    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # Forward
        optimizer.zero_grad()
        logits = model(images)  # (B, num_classes, H, W)
        loss = criterion(logits, masks)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch [{epoch}] Batch [{batch_idx+1}/{num_batches}] Loss: {loss.item():.4f}")
    
    avg_loss = running_loss / num_batches
    return avg_loss


def validate(model, dataloader, criterion, device, num_classes=21):
    """
    Validate model and compute loss + mIoU.
    
    Returns:
        dict with 'loss', 'miou', 'iou_per_class'
    """
    model.eval()
    running_loss = 0.0
    num_batches = len(dataloader)
    iou_metric = IoUMetric(num_classes=num_classes, ignore_index=255)
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            logits = model(images)  # (B, num_classes, H, W)
            loss = criterion(logits, masks)
            running_loss += loss.item()
            
            # Compute predictions
            preds = torch.argmax(logits, dim=1)  # (B, H, W)
            iou_metric.update(preds, masks)
    
    avg_loss = running_loss / num_batches
    iou_results = iou_metric.compute()
    
    return {
        'loss': avg_loss,
        'miou': iou_results['miou'],
        'iou_per_class': iou_results['iou_per_class']
    }


# ---------------------------
# Main training loop
# ---------------------------
def main():
    # Hyperparameters
    NUM_CLASSES = 21
    IMG_SIZE = (256, 256)
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10  # for early stopping
    
    # Paths
    CHECKPOINT_DIR = Path('./checkpoints')
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    BEST_MODEL_PATH = CHECKPOINT_DIR / 'best_model.pth'
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    
    # Dataloaders
    print("\nLoading datasets...")
    train_loader = get_voc_dataloader(
        root='./archive',
        image_set='train',
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=(device.type == 'cuda')
    )
    
    val_loader = get_voc_dataloader(
        root='./archive',
        image_set='val',
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=(device.type == 'cuda')
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Model
    print("\nInitializing model...")
    model = MBV3SmallSeg(
        num_classes=NUM_CLASSES,
        backbone_pretrained=True,
        input_size=IMG_SIZE,
        dropout=0.1
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    best_miou = 0.0
    epochs_without_improvement = 0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_results = validate(model, val_loader, criterion, device, num_classes=NUM_CLASSES)
        val_loss = val_results['loss']
        val_miou = val_results['miou']
        
        epoch_time = time.time() - epoch_start
        
        # Update scheduler
        scheduler.step(val_miou)
        
        # Print epoch summary
        print(f"\n{'='*60}")
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val mIoU:   {val_miou:.4f}")
        print(f"  Time:       {epoch_time:.1f}s")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_miou': val_miou,
                'val_loss': val_loss,
            }, BEST_MODEL_PATH)
            print(f"  ✓ New best mIoU! Model saved to {BEST_MODEL_PATH}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")
        
        print(f"{'='*60}\n")
        
        # Early stopping
        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs (patience={PATIENCE})")
            break
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation mIoU: {best_miou:.4f}")
    print(f"Best model saved at: {BEST_MODEL_PATH}")
    print("="*60)


if __name__ == '__main__':
    main()