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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import numpy as np
import time
from pathlib import Path
from torchsummary import summary
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


def check_model_sanity(model, dataloader, device, num_classes=21):
    """
    Perform a quick sanity check on model initialization.
    """
    model.eval()
    with torch.no_grad():
        # Get one batch
        images, masks = next(iter(dataloader))
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        
        # Check if predictions are reasonable
        unique_preds = torch.unique(preds)
        print(f"Model predicting {len(unique_preds)} unique classes: {unique_preds.cpu().numpy()}")
        
        # Check if all predictions are the same (bad initialization)
        if len(unique_preds) == 1:
            print("⚠️  WARNING: Model is predicting only one class - may need better initialization")
        
        # Check prediction distribution
        pred_dist = torch.bincount(preds.flatten(), minlength=num_classes)
        most_frequent = torch.argmax(pred_dist)
        percentage = (pred_dist[most_frequent] / preds.numel() * 100).item()
        print(f"Most frequent prediction: class {most_frequent} ({percentage:.1f}% of pixels)")
        
        if percentage > 90:
            print("⚠️  WARNING: Model heavily biased towards one class")
    
    model.train()  # Reset to training mode


def debug_training_step(model, dataloader, criterion, device, num_classes=21):
    """
    Debug a single training step to check gradients and learning.
    """
    model.train()
    
    # Get one batch
    images, masks = next(iter(dataloader))
    images = images.to(device)
    masks = masks.to(device)
    
    print(f"\n🔍 DEBUG: Training step analysis")
    print(f"Input shape: {images.shape}")
    print(f"Target shape: {masks.shape}")
    print(f"Target unique values: {torch.unique(masks).cpu().numpy()}")
    
    # Forward pass
    model.zero_grad()
    logits = model(images)
    loss = criterion(logits, masks)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Loss value: {loss.item():.6f}")
    
    # Check predictions before training
    with torch.no_grad():
        preds_before = torch.argmax(logits, dim=1)
        unique_preds_before = torch.unique(preds_before)
        print(f"Predictions before: {len(unique_preds_before)} classes {unique_preds_before.cpu().numpy()}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    total_grad_norm = 0
    param_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm ** 2
            param_count += 1
    
    total_grad_norm = total_grad_norm ** 0.5
    print(f"Total gradient norm: {total_grad_norm:.6f}")
    print(f"Parameters with gradients: {param_count}")
    
    if total_grad_norm < 1e-6:
        print("⚠️  WARNING: Very small gradients - model may not be learning!")
    
    return loss.item(), total_grad_norm


# ---------------------------
# Training and validation functions
# ---------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch):
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
        
        # Add gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights BEFORE updating learning rate
        optimizer.step()
        scheduler.step()  # OneCycleLR needs to be called every batch
        
        running_loss += loss.item()
        
        # Print progress every 25 batches or at the end
        if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == num_batches:
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
    IMG_SIZE = (256, 256)  # Revert - 256x256 worked better than 224x224
    BATCH_SIZE = 16  # Increase batch size for more stable gradients
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3  # Higher learning rate for better convergence
    WEIGHT_DECAY = 1e-4
    PATIENCE = 15  # More patience for convergence
    
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
    summary(model, input_size=(3, IMG_SIZE[0], IMG_SIZE[1]))
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
    # Use a simpler scheduler that starts with warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LEARNING_RATE, 
        epochs=NUM_EPOCHS, 
        steps_per_epoch=len(train_loader),
        pct_start=0.1  # 10% warmup
    )
    
    # Sanity check model initialization
    print("\nPerforming model sanity check...")
    check_model_sanity(model, val_loader, device, num_classes=NUM_CLASSES)
    
    # Debug first training step
    print("\nDebugging first training step...")
    debug_loss, debug_grad_norm = debug_training_step(model, train_loader, criterion, device, num_classes=NUM_CLASSES)
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    best_miou = 0.0
    epochs_without_improvement = 0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch)
        
        # Validate
        val_results = validate(model, val_loader, criterion, device, num_classes=NUM_CLASSES)
        val_loss = val_results['loss']
        val_miou = val_results['miou']
        
        epoch_time = time.time() - epoch_start
        
        # OneCycleLR is updated per batch, no need to call here
        
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