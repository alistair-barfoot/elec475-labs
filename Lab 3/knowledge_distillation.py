"""
Knowledge Distillation Training for MobileNet Semantic Segmentation

This script implements knowledge distillation using FCN-ResNet50 as the teacher model
and MBV3SmallSeg as the student model. The distillation loss combines:
- α * H(y, σ(z_s; T=1)) - standard cross-entropy with ground truth
- β * H(σ(z_t; T=τ), σ(z_s; T=τ)) - distillation loss between teacher and student outputs

Features:
- Teacher model (FCN-ResNet50) with pretrained weights
- Student model (MBV3SmallSeg) 
- Temperature-scaled softmax for knowledge distillation
- Combined loss function with configurable α and β weights
- Validation with mIoU metric
- Model checkpointing and early stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
from pathlib import Path

from MobileNet_model import MBV3SmallSeg
from dataloader import get_voc_dataloader
from torchvision.models.segmentation import fcn_resnet50


# ---------------------------
# Knowledge Distillation Loss
# ---------------------------
class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining:
    - α * H(y, σ(z_s; T=1)) - standard cross-entropy with ground truth
    - β * H(σ(z_t; T=τ), σ(z_s; T=τ)) - distillation loss between teacher and student
    
    Args:
        alpha: weight for ground truth loss (typically 0.3-0.7)
        beta: weight for distillation loss (typically 0.3-0.7, should sum to 1 with alpha)
        temperature: temperature for knowledge distillation (typically 3-5)
        ignore_index: index to ignore in ground truth loss (255 for VOC)
    """
    def __init__(self, alpha=0.5, beta=0.5, temperature=4.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.ignore_index = ignore_index
        
        # Standard cross-entropy for ground truth
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
    def forward(self, student_logits, teacher_logits, targets):
        """
        Args:
            student_logits: (B, C, H, W) - student model output
            teacher_logits: (B, C, H, W) - teacher model output  
            targets: (B, H, W) - ground truth labels
            
        Returns:
            total_loss: combined distillation loss
            ce_loss: cross-entropy component 
            kd_loss: knowledge distillation component
        """
        # Standard cross-entropy loss with ground truth (T=1)
        ce_loss = self.ce_loss(student_logits, targets)
        
        # Knowledge distillation loss (T=τ)
        # Apply temperature scaling and softmax
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # KL divergence between teacher and student
        # Use 'mean' reduction for spatial dimensions to match CE loss scale
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction='mean')
        
        # Scale by T² to maintain gradient magnitude
        kd_loss = kd_loss * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * ce_loss + self.beta * kd_loss
        
        return total_loss, ce_loss, kd_loss


# ---------------------------
# Teacher Model Wrapper
# ---------------------------
class TeacherModel(nn.Module):
    """
    FCN-ResNet50 teacher model wrapper that matches student output format.
    """
    def __init__(self, num_classes=21, pretrained=True):
        super().__init__()
        
        # Load FCN-ResNet50 with pretrained weights
        if pretrained:
            try:
                from torchvision.models.segmentation import FCN_ResNet50_Weights
                weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
                self.model = fcn_resnet50(weights=weights)
            except:
                self.model = fcn_resnet50(pretrained=True)
        else:
            self.model = fcn_resnet50(pretrained=False)
            
        # Modify classifier if number of classes doesn't match
        if self.model.classifier[4].out_channels != num_classes:
            self.model.classifier[4] = nn.Conv2d(
                self.model.classifier[4].in_channels, 
                num_classes, 
                kernel_size=1
            )
        
        # Freeze teacher model parameters
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        """
        Forward pass returning logits in same format as student.
        """
        output = self.model(x)
        return output['out']  # FCN returns dict with 'out' key


# ---------------------------
# mIoU metric computation (reused from training_loop.py)
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
def train_one_epoch_kd(student_model, teacher_model, dataloader, criterion, optimizer, device, epoch):
    """
    Train student model for one epoch with knowledge distillation.
    
    Returns:
        dict with avg losses (total, ce, kd)
    """
    student_model.train()
    teacher_model.eval()
    
    running_total_loss = 0.0
    running_ce_loss = 0.0
    running_kd_loss = 0.0
    num_batches = len(dataloader)
    
    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass - student
        student_logits = student_model(images)  # (B, num_classes, H, W)
        
        # Forward pass - teacher (no gradients)
        with torch.no_grad():
            teacher_logits = teacher_model(images)
            
            # Resize teacher output to match student if needed
            if teacher_logits.shape != student_logits.shape:
                teacher_logits = F.interpolate(
                    teacher_logits, 
                    size=student_logits.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
        
        # Compute distillation loss
        total_loss, ce_loss, kd_loss = criterion(student_logits, teacher_logits, masks)
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        running_total_loss += total_loss.item()
        running_ce_loss += ce_loss.item()
        running_kd_loss += kd_loss.item()
        
        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch [{epoch}] Batch [{batch_idx+1}/{num_batches}] "
                  f"Total: {total_loss.item():.4f} CE: {ce_loss.item():.4f} KD: {kd_loss.item():.4f}")
    
    return {
        'total_loss': running_total_loss / num_batches,
        'ce_loss': running_ce_loss / num_batches,
        'kd_loss': running_kd_loss / num_batches
    }


def validate_kd(student_model, teacher_model, dataloader, criterion, device, num_classes=21):
    """
    Validate student model and compute loss + mIoU.
    
    Returns:
        dict with losses and metrics
    """
    student_model.eval()
    teacher_model.eval()
    
    running_total_loss = 0.0
    running_ce_loss = 0.0
    running_kd_loss = 0.0
    num_batches = len(dataloader)
    iou_metric = IoUMetric(num_classes=num_classes, ignore_index=255)
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            # Forward pass
            student_logits = student_model(images)
            teacher_logits = teacher_model(images)
            
            # Resize teacher output if needed
            if teacher_logits.shape != student_logits.shape:
                teacher_logits = F.interpolate(
                    teacher_logits, 
                    size=student_logits.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Compute losses
            total_loss, ce_loss, kd_loss = criterion(student_logits, teacher_logits, masks)
            running_total_loss += total_loss.item()
            running_ce_loss += ce_loss.item()
            running_kd_loss += kd_loss.item()
            
            # Compute predictions for mIoU
            preds = torch.argmax(student_logits, dim=1)  # (B, H, W)
            iou_metric.update(preds, masks)
    
    iou_results = iou_metric.compute()
    
    return {
        'total_loss': running_total_loss / num_batches,
        'ce_loss': running_ce_loss / num_batches,
        'kd_loss': running_kd_loss / num_batches,
        'miou': iou_results['miou'],
        'iou_per_class': iou_results['iou_per_class']
    }

def train_knowledge_distillation(teacher, student, train_loader, val_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device, save_path='best_student_kd.pth'):
    """
    Integrated knowledge distillation training for segmentation models.
    
    Args:
        teacher: Teacher model (FCN-ResNet50)
        student: Student model (MBV3SmallSeg) 
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        T: Temperature for knowledge distillation
        soft_target_loss_weight: Weight for distillation loss (β)
        ce_loss_weight: Weight for ground truth loss (α)
        device: Device to train on
        save_path: Path to save best model
    """
    # Loss functions
    ce_loss = nn.CrossEntropyLoss(ignore_index=255)  # Ignore boundary pixels
    optimizer = optim.AdamW(student.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode
    
    # Tracking
    best_miou = 0.0
    epochs_without_improvement = 0
    patience = 10
    
    print(f"\nStarting Knowledge Distillation Training...")
    print(f"Temperature: {T}, CE Weight: {ce_loss_weight}, KD Weight: {soft_target_loss_weight}")
    print("="*80)

    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        running_ce_loss = 0.0
        running_kd_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients
            with torch.no_grad():
                teacher_logits = teacher(inputs)
                
                # Resize teacher output to match student if needed  
                if teacher_logits.shape != inputs.shape:
                    teacher_logits = F.interpolate(
                        teacher_logits, 
                        size=inputs.shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )

            # Forward pass with the student model
            student_logits = student(inputs)
            
            # Ensure teacher and student outputs have same spatial dimensions
            if teacher_logits.shape != student_logits.shape:
                teacher_logits = F.interpolate(
                    teacher_logits, 
                    size=student_logits.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )

            # Soften the logits by applying temperature scaling
            soft_targets = F.softmax(teacher_logits / T, dim=1)
            soft_student = F.log_softmax(student_logits / T, dim=1)

            # Calculate the soft targets loss (KL divergence)
            # Use mean reduction to match segmentation scale
            soft_targets_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean') * (T**2)

            # Calculate the true label loss  
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_ce_loss += label_loss.item()
            running_kd_loss += soft_targets_loss.item()
            
            # Print progress
            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch [{epoch+1}] Batch [{batch_idx+1}/{num_batches}] "
                      f"Total: {loss.item():.4f} CE: {label_loss.item():.4f} KD: {soft_targets_loss.item():.4f}")

        # Validation
        student.eval()
        val_miou = validate_segmentation_model(student, val_loader, device)
        
        # Update scheduler
        scheduler.step(val_miou)
        
        # Print epoch summary
        avg_loss = running_loss / len(train_loader)
        avg_ce = running_ce_loss / len(train_loader) 
        avg_kd = running_kd_loss / len(train_loader)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train - Total: {avg_loss:.4f} CE: {avg_ce:.4f} KD: {avg_kd:.4f}")
        print(f"  Val mIoU: {val_miou:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            epochs_without_improvement = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_miou': val_miou,
                'distillation_params': {
                    'temperature': T,
                    'ce_weight': ce_loss_weight,
                    'kd_weight': soft_target_loss_weight
                }
            }, save_path)
            print(f"  ✓ New best mIoU! Model saved to {save_path}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")
        
        print("-" * 60)
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print(f"\nTraining complete! Best mIoU: {best_miou:.4f}")
    return best_miou


def validate_segmentation_model(model, val_loader, device, num_classes=21):
    """
    Validate segmentation model and compute mIoU.
    """
    model.eval()
    iou_metric = IoUMetric(num_classes=num_classes, ignore_index=255)
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            iou_metric.update(preds, masks)
    
    results = iou_metric.compute()
    return results['miou']


# ---------------------------
# Main knowledge distillation training
# ---------------------------
def main():
    # Hyperparameters
    NUM_CLASSES = 21
    IMG_SIZE = (256, 256)
    BATCH_SIZE = 8  # Reduced for memory efficiency with teacher model
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10
    
    # Knowledge distillation parameters
    ALPHA = 0.75  # Weight for ground truth loss (ce_loss_weight)
    BETA = 0.25   # Weight for distillation loss (soft_target_loss_weight) 
    TEMPERATURE = 4.0  # Temperature for knowledge distillation
    
    # Paths
    CHECKPOINT_DIR = Path('./checkpoints_kd')
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    BEST_MODEL_PATH = CHECKPOINT_DIR / 'best_student_kd.pth'
    
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
        num_workers=2 if device.type == 'cuda' else 0,  # Reduced for memory
        pin_memory=(device.type == 'cuda')
    )
    
    val_loader = get_voc_dataloader(
        root='./archive',
        image_set='val',
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2 if device.type == 'cuda' else 0,
        pin_memory=(device.type == 'cuda')
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Models
    print("\nInitializing models...")
    
    # Teacher model (FCN-ResNet50)
    print("Loading teacher model (FCN-ResNet50)...")
    teacher_model = TeacherModel(num_classes=NUM_CLASSES, pretrained=True)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()  # Always in eval mode
    
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    print(f"Teacher params: {teacher_params:,}")
    
    # Student model (MBV3SmallSeg)
    print("Loading student model (MBV3SmallSeg)...")
    student_model = MBV3SmallSeg(
        num_classes=NUM_CLASSES,
        backbone_pretrained=True,
        input_size=IMG_SIZE,
        dropout=0.1
    )
    
    # Load pre-trained weights from best_model.pth if available
    pretrained_model_path = Path('best_model.pth')
    if pretrained_model_path.exists():
        print(f"Loading pre-trained student weights from {pretrained_model_path}...")
        try:
            checkpoint = torch.load(pretrained_model_path, map_location='cpu', weights_only=False)
            student_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Successfully loaded pre-trained weights (epoch {checkpoint.get('epoch', 'unknown')})")
            print(f"  Pre-trained validation mIoU: {checkpoint.get('val_miou', 'unknown')}")
        except Exception as e:
            print(f"⚠️  Could not load pre-trained weights: {e}")
            print("  Continuing with randomly initialized student model...")
    else:
        print(f"⚠️  Pre-trained model not found at {pretrained_model_path}")
        print("  Starting with randomly initialized student model...")
    
    student_model = student_model.to(device)
    
    student_total_params = sum(p.numel() for p in student_model.parameters())
    student_trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    print(f"Student total params: {student_total_params:,}")
    print(f"Student trainable params: {student_trainable_params:,}")
    
    # Run integrated knowledge distillation training
    print("\n" + "="*80)
    print("Starting Integrated Knowledge Distillation Training...")
    print(f"Teacher: FCN-ResNet50 ({teacher_params:,} params)")
    print(f"Student: MBV3SmallSeg ({student_trainable_params:,} trainable params)")
    print(f"CE Weight: {ALPHA}, KD Weight: {BETA}, Temperature: {TEMPERATURE}")
    print("="*80)
    
    # Train using the integrated function
    best_miou = train_knowledge_distillation(
        teacher=teacher_model,
        student=student_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        T=TEMPERATURE,
        soft_target_loss_weight=BETA,  # β for distillation loss
        ce_loss_weight=ALPHA,         # α for ground truth loss
        device=device,
        save_path=str(BEST_MODEL_PATH)
    )
    
    print("\n" + "="*80)
    print("Knowledge Distillation Training Complete!")
    print(f"Best validation mIoU: {best_miou:.4f}")
    print(f"Best student model saved at: {BEST_MODEL_PATH}")
    print("="*80)


if __name__ == '__main__':
    main()
