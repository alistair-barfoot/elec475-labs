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
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from MobileNet_model import MBV3SmallSeg
from dataloader import get_voc_dataloader
from torchvision.models.segmentation import fcn_resnet50


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Knowledge Distillation Training for MobileNet Semantic Segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Knowledge Distillation Methods:
  -r, --response    Response-based distillation using output logits (temperature scaling)
  -f, --feature     Feature-based distillation using intermediate representations (cosine similarity)

Examples:
  python knowledge_distillation.py -r    # Response-based distillation (default)
  python knowledge_distillation.py -f    # Feature-based distillation
        """
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-r', '--response', 
                       action='store_true',
                       help='Use response-based knowledge distillation (temperature scaling)')
    
    group.add_argument('-f', '--feature',
                       action='store_true',
                       help='Use feature-based knowledge distillation (cosine similarity)')
    
    parser.add_argument('--epochs',
                        type=int,
                        default=50,
                        help='Number of training epochs (default: 50)')
    
    parser.add_argument('--batch-size',
                        type=int,
                        default=8,
                        help='Batch size for training (default: 8)')
    
    parser.add_argument('--learning-rate',
                        type=float,
                        default=1e-3,
                        help='Learning rate (default: 1e-3)')
    
    parser.add_argument('--temperature',
                        type=float,
                        default=4.0,
                        help='Temperature for response-based distillation (default: 4.0)')
    
    parser.add_argument('--alpha',
                        type=float,
                        default=0.75,
                        help='Weight for ground truth loss (default: 0.75)')
    
    parser.add_argument('--beta',
                        type=float,
                        default=0.25,
                        help='Weight for distillation loss (default: 0.25)')
    
    parser.add_argument('--save-plots',
                        action='store_true',
                        help='Save training/validation plots (default: False)')
    
    parser.add_argument('--no-warm-start',
                        action='store_true',
                        help='Disable warm start from best_model.pth (default: warm start enabled)')
    
    parser.add_argument('--plot-suffix',
                        type=str,
                        default='',
                        help='Suffix to add to plot filenames (e.g., "--plot-suffix exp1" creates plots with "_exp1" suffix)')
    
    parser.add_argument('--plot-prefix',
                        type=str,
                        default='',
                        help='Prefix to add to plot filenames (e.g., "--plot-prefix warmstart" creates plots with "warmstart_" prefix)')
    
    args = parser.parse_args()
    
    # If no method specified, default to response-based
    if not args.response and not args.feature:
        args.response = True
    
    return args


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
                weights = FCN_ResNet50_Weights.DEFAULT
                self.model = fcn_resnet50(weights=weights)
            except:
                self.model = fcn_resnet50(weights=weights, progress=True)
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
            print(f"  Epoch [{epoch+1}] Batch [{batch_idx+1}/{num_batches}] "
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

def train_knowledge_distillation(teacher, student, train_loader, val_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device, save_path='best_student_kd.pth', save_plots=False, method_name='response-based', plot_prefix='', plot_suffix=''):
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
        save_plots: Whether to save training plots
        method_name: Name of distillation method for plots
        plot_prefix: Prefix for plot filenames
        plot_suffix: Suffix for plot filenames
    """
    # Loss functions
    ce_loss = nn.CrossEntropyLoss(ignore_index=255)  # Ignore boundary pixels
    optimizer = optim.AdamW(student.parameters(), lr=learning_rate, weight_decay=5e-4)  # Increased regularization
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)  # More aggressive LR reduction
    
    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode
    
    # Tracking
    best_miou = 0.0
    epochs_without_improvement = 0
    patience = 10
    
    # Lists to track training history
    train_losses = {'total': [], 'ce': [], 'kd': []}
    val_losses = {'total': [], 'ce': [], 'kd': []}
    val_miou_history = []
    epoch_times = []
    
    print(f"\nStarting Knowledge Distillation Training...")
    print(f"Temperature: {T}, CE Weight: {ce_loss_weight}, KD Weight: {soft_target_loss_weight}")
    print("="*80)
    
    # Start overall training timer
    training_start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
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
        
        # Compute validation losses
        student.eval()
        val_total_loss = 0.0
        val_ce_loss = 0.0
        val_kd_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Teacher forward pass
                teacher_logits = teacher(inputs)
                if teacher_logits.shape != inputs.shape:
                    teacher_logits = F.interpolate(
                        teacher_logits, 
                        size=inputs.shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # Student forward pass
                student_logits = student(inputs)
                
                if teacher_logits.shape != student_logits.shape:
                    teacher_logits = F.interpolate(
                        teacher_logits, 
                        size=student_logits.shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # Compute validation losses
                soft_targets = F.softmax(teacher_logits / T, dim=1)
                soft_student = F.log_softmax(student_logits / T, dim=1)
                soft_targets_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean') * (T**2)
                label_loss = ce_loss(student_logits, labels)
                total_loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss
                
                val_total_loss += total_loss.item()
                val_ce_loss += label_loss.item()
                val_kd_loss += soft_targets_loss.item()
                val_batches += 1
        
        # Update scheduler
        scheduler.step(val_miou)
        
        # Calculate training averages
        avg_loss = running_loss / len(train_loader)
        avg_ce = running_ce_loss / len(train_loader) 
        avg_kd = running_kd_loss / len(train_loader)
        
        # Record losses and metrics
        train_losses['total'].append(avg_loss)
        train_losses['ce'].append(avg_ce)
        train_losses['kd'].append(avg_kd)
        
        val_losses['total'].append(val_total_loss / val_batches)
        val_losses['ce'].append(val_ce_loss / val_batches)
        val_losses['kd'].append(val_kd_loss / val_batches)
        
        val_miou_history.append(val_miou)
        
        # Calculate epoch timing
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        
        # Estimate remaining time
        avg_epoch_time = np.mean(epoch_times)
        remaining_epochs = epochs - (epoch + 1)
        estimated_time_remaining = remaining_epochs * avg_epoch_time
        
        # Format time strings
        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            if hours > 0:
                return f"{hours:02d}h {minutes:02d}m {secs:02d}s"
            elif minutes > 0:
                return f"{minutes:02d}m {secs:02d}s"
            else:
                return f"{secs:02d}s"
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs} [{format_time(epoch_duration)}]")
        print(f"  Train - Total: {avg_loss:.4f} CE: {avg_ce:.4f} KD: {avg_kd:.4f}")
        print(f"  Val   - Total: {val_total_loss/val_batches:.4f} CE: {val_ce_loss/val_batches:.4f} KD: {val_kd_loss/val_batches:.4f}")
        print(f"  Val mIoU: {val_miou:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        if remaining_epochs > 0:
            print(f"  ETA: {format_time(estimated_time_remaining)} (avg {format_time(avg_epoch_time)}/epoch)")
        
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
    
    # Calculate total training time
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}h {minutes:02d}m {secs:02d}s"
        elif minutes > 0:
            return f"{minutes:02d}m {secs:02d}s"
        else:
            return f"{secs:02d}s"
    
    print(f"\nTraining complete! Best mIoU: {best_miou:.4f}")
    print(f"Total training time: {format_time(total_training_time)}")
    if epoch_times:
        print(f"Average time per epoch: {format_time(np.mean(epoch_times))}")
        print(f"Fastest epoch: {format_time(np.min(epoch_times))}")
        print(f"Slowest epoch: {format_time(np.max(epoch_times))}")
    
    # Create training plots if requested
    if save_plots:
        print("\nGenerating training plots...")
        plot_training_history(
            train_losses=train_losses,
            val_losses=val_losses,
            train_miou=[],  # No train mIoU computed during training for efficiency
            val_miou=val_miou_history,
            method_name=method_name,
            save_dir='./plots',
            plot_prefix=plot_prefix,
            plot_suffix=plot_suffix
        )
    
    return best_miou

def train_cosine_loss(teacher, student, train_loader, val_loader, epochs, learning_rate, hidden_rep_loss_weight, ce_loss_weight, device, save_path='best_student_feature_kd.pth', save_plots=False, method_name='feature-based', plot_prefix="", plot_suffix=""):
    """
    Feature-based knowledge distillation using cosine similarity between hidden representations.
    
    Args:
        teacher: Teacher model (FCN-ResNet50)
        student: Student model (MBV3SmallSeg) 
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        hidden_rep_loss_weight: Weight for cosine similarity loss
        ce_loss_weight: Weight for ground truth loss
        device: Device to train on
        save_path: Path to save best model
        save_plots: Whether to save training plots
        method_name: Name of distillation method for plots
    """
    ce_loss = nn.CrossEntropyLoss(ignore_index=255)
    cosine_loss = nn.CosineEmbeddingLoss()
    optimizer = optim.AdamW(student.parameters(), lr=learning_rate, weight_decay=5e-4)  # Increased regularization
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)  # More aggressive LR reduction

    teacher.to(device)
    student.to(device)
    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode
    
    # Tracking
    best_miou = 0.0
    epochs_without_improvement = 0
    patience = 10
    
    # Lists to track training history
    train_losses = {'total': [], 'ce': [], 'feature': []}
    val_losses = {'total': [], 'ce': [], 'feature': []}
    val_miou_history = []
    epoch_times = []
    
    print(f"\nStarting Feature-based Knowledge Distillation Training...")
    print(f"CE Weight: {ce_loss_weight}, Feature Weight: {hidden_rep_loss_weight}")
    print("="*80)
    
    # Start overall training timer
    training_start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        student.train()
        running_loss = 0.0
        running_ce_loss = 0.0
        running_feature_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model and get hidden representation
            with torch.no_grad():
                teacher_logits = teacher(inputs)
                # For segmentation, we'll use the teacher's output as the "hidden representation"
                # In a more sophisticated approach, you would extract intermediate features
                teacher_hidden = teacher_logits.view(teacher_logits.size(0), -1)  # Flatten spatial dims

            # Forward pass with the student model
            student_logits = student(inputs)
            
            # Ensure student and teacher outputs have same spatial dimensions
            if teacher_logits.shape != student_logits.shape:
                teacher_logits = F.interpolate(
                    teacher_logits, 
                    size=student_logits.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
                teacher_hidden = teacher_logits.view(teacher_logits.size(0), -1)
            
            student_hidden = student_logits.view(student_logits.size(0), -1)  # Flatten spatial dims

            # Calculate the cosine loss
            # Target is a vector of ones (maximize cosine similarity)
            hidden_rep_loss = cosine_loss(
                student_hidden, 
                teacher_hidden, 
                target=torch.ones(inputs.size(0)).to(device)
            )

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = hidden_rep_loss_weight * hidden_rep_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_ce_loss += label_loss.item()
            running_feature_loss += hidden_rep_loss.item()
            
            # Print progress
            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch [{epoch+1}] Batch [{batch_idx+1}/{num_batches}] "
                      f"Total: {loss.item():.4f} CE: {label_loss.item():.4f} Feature: {hidden_rep_loss.item():.4f}")

        # Validation
        student.eval()
        val_miou = validate_segmentation_model(student, val_loader, device)
        
        # Compute validation losses
        student.eval()
        val_total_loss = 0.0
        val_ce_loss = 0.0
        val_feature_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Teacher forward pass
                teacher_logits = teacher(inputs)
                teacher_hidden = teacher_logits.view(teacher_logits.size(0), -1)
                
                # Student forward pass
                student_logits = student(inputs)
                
                # Ensure teacher and student outputs have same spatial dimensions
                if teacher_logits.shape != student_logits.shape:
                    teacher_logits = F.interpolate(
                        teacher_logits, 
                        size=student_logits.shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                    teacher_hidden = teacher_logits.view(teacher_logits.size(0), -1)
                
                student_hidden = student_logits.view(student_logits.size(0), -1)
                
                # Compute validation losses
                hidden_rep_loss = cosine_loss(
                    student_hidden, 
                    teacher_hidden, 
                    target=torch.ones(inputs.size(0)).to(device)
                )
                label_loss = ce_loss(student_logits, labels)
                total_loss = hidden_rep_loss_weight * hidden_rep_loss + ce_loss_weight * label_loss
                
                val_total_loss += total_loss.item()
                val_ce_loss += label_loss.item()
                val_feature_loss += hidden_rep_loss.item()
                val_batches += 1
        
        # Update scheduler
        scheduler.step(val_miou)
        
        # Calculate training averages
        avg_loss = running_loss / len(train_loader)
        avg_ce = running_ce_loss / len(train_loader) 
        avg_feature = running_feature_loss / len(train_loader)
        
        # Record losses and metrics
        train_losses['total'].append(avg_loss)
        train_losses['ce'].append(avg_ce)
        train_losses['feature'].append(avg_feature)
        
        val_losses['total'].append(val_total_loss / val_batches)
        val_losses['ce'].append(val_ce_loss / val_batches)
        val_losses['feature'].append(val_feature_loss / val_batches)
        
        val_miou_history.append(val_miou)
        
        # Calculate epoch timing
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        
        # Estimate remaining time
        avg_epoch_time = np.mean(epoch_times)
        remaining_epochs = epochs - (epoch + 1)
        estimated_time_remaining = remaining_epochs * avg_epoch_time
        
        # Format time strings
        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            if hours > 0:
                return f"{hours:02d}h {minutes:02d}m {secs:02d}s"
            elif minutes > 0:
                return f"{minutes:02d}m {secs:02d}s"
            else:
                return f"{secs:02d}s"
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs} [{format_time(epoch_duration)}]")
        print(f"  Train - Total: {avg_loss:.4f} CE: {avg_ce:.4f} Feature: {avg_feature:.4f}")
        print(f"  Val   - Total: {val_total_loss/val_batches:.4f} CE: {val_ce_loss/val_batches:.4f} Feature: {val_feature_loss/val_batches:.4f}")
        print(f"  Val mIoU: {val_miou:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        if remaining_epochs > 0:
            print(f"  ETA: {format_time(estimated_time_remaining)} (avg {format_time(avg_epoch_time)}/epoch)")
        
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
                    'method': 'feature-based',
                    'ce_weight': ce_loss_weight,
                    'feature_weight': hidden_rep_loss_weight
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
    
    # Calculate total training time
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}h {minutes:02d}m {secs:02d}s"
        elif minutes > 0:
            return f"{minutes:02d}m {secs:02d}s"
        else:
            return f"{secs:02d}s"
    
    print(f"\nFeature-based training complete! Best mIoU: {best_miou:.4f}")
    print(f"Total training time: {format_time(total_training_time)}")
    if epoch_times:
        print(f"Average time per epoch: {format_time(np.mean(epoch_times))}")
        print(f"Fastest epoch: {format_time(np.min(epoch_times))}")
        print(f"Slowest epoch: {format_time(np.max(epoch_times))}")
    
    # Create training plots if requested
    if save_plots:
        print("\nGenerating training plots...")
        plot_training_history(
            train_losses=train_losses,
            val_losses=val_losses,
            train_miou=[],  # No train mIoU computed during training for efficiency
            val_miou=val_miou_history,
            method_name=method_name,
            save_dir='./plots',
            plot_prefix=plot_prefix,
            plot_suffix=plot_suffix
        )
    
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
# Plotting utilities
# ---------------------------
def plot_training_history(train_losses, val_losses, train_miou, val_miou, method_name, save_dir='./plots', plot_prefix='', plot_suffix=''):
    """
    Plot training and validation losses and mIoU over epochs.
    
    Args:
        train_losses: dict with keys 'total', 'ce', 'kd' containing lists of training losses
        val_losses: dict with keys 'total', 'ce', 'kd' containing lists of validation losses  
        train_miou: list of training mIoU values per epoch
        val_miou: list of validation mIoU values per epoch
        method_name: name of distillation method for plot title
        save_dir: directory to save plots
        plot_prefix: prefix for plot filenames
        plot_suffix: suffix for plot filenames
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    epochs = range(1, len(val_miou) + 1)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Knowledge Distillation Training History ({method_name})', fontsize=16, fontweight='bold')
    
    # Plot 1: Total Loss
    ax1.plot(epochs, train_losses['total'], 'b-', label='Train Total Loss', linewidth=2)
    ax1.plot(epochs, val_losses['total'], 'r-', label='Val Total Loss', linewidth=2)
    ax1.set_title('Total Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cross-Entropy Loss
    ax2.plot(epochs, train_losses['ce'], 'b-', label='Train CE Loss', linewidth=2)
    ax2.plot(epochs, val_losses['ce'], 'r-', label='Val CE Loss', linewidth=2)
    ax2.set_title('Cross-Entropy Loss', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('CE Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distillation Loss (KD or Feature)
    kd_label = 'KD Loss' if 'kd' in train_losses else 'Feature Loss'
    kd_key = 'kd' if 'kd' in train_losses else 'feature'
    ax3.plot(epochs, train_losses[kd_key], 'b-', label=f'Train {kd_label}', linewidth=2)
    ax3.plot(epochs, val_losses[kd_key], 'r-', label=f'Val {kd_label}', linewidth=2)
    ax3.set_title(f'{kd_label}', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel(f'{kd_label}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: mIoU
    if train_miou:  # Only plot train mIoU if available
        ax4.plot(epochs, train_miou, 'b-', label='Train mIoU', linewidth=2)
    ax4.plot(epochs, val_miou, 'r-', label='Val mIoU', linewidth=2)
    ax4.set_title('Mean IoU', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('mIoU')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Highlight best validation mIoU
    best_epoch = np.argmax(val_miou) + 1
    best_miou = max(val_miou)
    ax4.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7)
    ax4.annotate(f'Best: {best_miou:.3f}\n@Epoch {best_epoch}', 
                xy=(best_epoch, best_miou), 
                xytext=(best_epoch + len(epochs)*0.1, best_miou),
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                fontsize=10, color='green', fontweight='bold')
    
    plt.tight_layout()
    
    # Generate plot filename with prefix and suffix
    base_filename = f'kd_training_history_{method_name.replace("-", "_")}'
    if plot_prefix:
        base_filename = f'{plot_prefix}_{base_filename}'
    if plot_suffix:
        base_filename = f'{base_filename}_{plot_suffix}'
    
    # Save plot
    plot_filename = f'{base_filename}.png'
    plot_path = save_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to: {plot_path}")
    
    # Also save a simple loss comparison plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses['total'], 'b-', label='Train Total', linewidth=2)
    plt.plot(epochs, val_losses['total'], 'r-', label='Val Total', linewidth=2)
    plt.title(f'Total Loss ({method_name})', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    if train_miou:
        plt.plot(epochs, train_miou, 'b-', label='Train mIoU', linewidth=2)
    plt.plot(epochs, val_miou, 'r-', label='Val mIoU', linewidth=2)
    plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best ({best_miou:.3f})')
    plt.title(f'Mean IoU ({method_name})', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Generate simple plot filename with prefix and suffix
    simple_base_filename = f'kd_loss_miou_{method_name.replace("-", "_")}'
    if plot_prefix:
        simple_base_filename = f'{plot_prefix}_{simple_base_filename}'
    if plot_suffix:
        simple_base_filename = f'{simple_base_filename}_{plot_suffix}'
    
    # Save simple plot
    simple_plot_filename = f'{simple_base_filename}.png'
    simple_plot_path = save_dir / simple_plot_filename
    plt.savefig(simple_plot_path, dpi=300, bbox_inches='tight')
    print(f"Loss/mIoU plot saved to: {simple_plot_path}")
    
    plt.show()  # Display plots
    plt.close('all')  # Close figures to free memory


# ---------------------------
# Main knowledge distillation training
# ---------------------------
def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Hyperparameters
    NUM_CLASSES = 21
    IMG_SIZE = (256, 256)
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10
    
    # Knowledge distillation parameters
    ALPHA = args.alpha    # Weight for ground truth loss
    BETA = args.beta      # Weight for distillation loss
    TEMPERATURE = args.temperature  # Temperature for response-based distillation
    
    # Determine distillation method and setup paths
    if args.response:
        method_name = "response-based"
        save_filename = 'best_student_response_kd.pth'
        print(f"Selected method: Response-based distillation (temperature scaling)")
    else:  # args.feature
        method_name = "feature-based"
        save_filename = 'best_student_feature_kd.pth'
        print(f"Selected method: Feature-based distillation (cosine similarity)")
    
    # Paths
    CHECKPOINT_DIR = Path('./checkpoints_kd')
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    BEST_MODEL_PATH = CHECKPOINT_DIR / save_filename
    
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
        num_workers=2 if device.type == 'cuda' else 0,
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
    
    # Load pre-trained segmentation weights for warm start (RECOMMENDED for overfitting prevention)
    pretrained_model_path = Path('best_model.pth')
    if pretrained_model_path.exists() and not args.no_warm_start:
        print(f"🔥 WARM START: Loading pre-trained segmentation weights from {pretrained_model_path}...")
        try:
            checkpoint = torch.load(pretrained_model_path, map_location='cpu', weights_only=False)
            student_model.load_state_dict(checkpoint['model_state_dict'])
            baseline_miou = checkpoint.get('val_miou', 'unknown')
            baseline_epoch = checkpoint.get('epoch', 'unknown')
            print(f"✅ Warm start successful!")
            print(f"   Starting from: {baseline_miou:.4f} mIoU (epoch {baseline_epoch})")
            print(f"   Benefits: Faster convergence, reduced overfitting risk")
        except Exception as e:
            print(f"⚠️  Could not load warm start weights: {e}")
            print("   Continuing with backbone-only initialization...")
    else:
        if args.no_warm_start:
            print(f"🚫 Warm start disabled by --no-warm-start flag")
        else:
            print(f"ℹ️  No warm start model found at {pretrained_model_path}")
        print("   Starting with backbone pretrained weights only")
    print("  Segmentation head randomly initialized for knowledge distillation")
    
    student_model = student_model.to(device)
    
    student_total_params = sum(p.numel() for p in student_model.parameters())
    student_trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    print(f"Student total params: {student_total_params:,}")
    print(f"Student trainable params: {student_trainable_params:,}")
    
    # Run knowledge distillation training based on selected method
    print("\n" + "="*80)
    print(f"Starting {method_name.title()} Knowledge Distillation Training...")
    print(f"Teacher: FCN-ResNet50 ({teacher_params:,} params)")
    print(f"Student: MBV3SmallSeg ({student_trainable_params:,} trainable params)")
    print(f"CE Weight: {ALPHA}, Distillation Weight: {BETA}")
    if args.response:
        print(f"Temperature: {TEMPERATURE}")
    print("="*80)
    
    # Choose training function based on method
    if args.response:
        # Response-based distillation
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
            save_path=str(BEST_MODEL_PATH),
            save_plots=args.save_plots,
            method_name=method_name,
            plot_prefix=args.plot_prefix,
            plot_suffix=args.plot_suffix
        )
    else:
        # Feature-based distillation
        best_miou = train_cosine_loss(
            teacher=teacher_model,
            student=student_model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            hidden_rep_loss_weight=BETA,  # β for feature similarity loss
            ce_loss_weight=ALPHA,         # α for ground truth loss
            device=device,
            save_path=str(BEST_MODEL_PATH),
            save_plots=args.save_plots,
            method_name=method_name,
            plot_prefix=args.plot_prefix,
            plot_suffix=args.plot_suffix
        )
    
    print("\n" + "="*80)
    print(f"{method_name.title()} Knowledge Distillation Training Complete!")
    print(f"Best validation mIoU: {best_miou:.4f}")
    print(f"Best student model saved at: {BEST_MODEL_PATH}")
    print("="*80)


if __name__ == '__main__':
    main()
