"""
Test script for MBV3SmallSeg model testing on VOC 2012.

This script:
- Loads the best trained model from checkpoints/best_model.pth
- Evaluates on test set with mIoU and per-class IoU metrics
- Visualizes sample predictions
- Saves evaluation results and sample images
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from PIL import Image
import torchvision.transforms as T

from MobileNet_model import MBV3SmallSeg
from dataloader import get_voc_dataloader, VOC_CLASSES
from training_loop import IoUMetric


def load_trained_model(checkpoint_path, num_classes=21, device='cpu'):
    """
    Load the trained MBV3SmallSeg model from checkpoint.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        num_classes: Number of segmentation classes
        device: Device to load the model on
    
    Returns:
        model: Loaded model in eval mode
        checkpoint_info: Dictionary with training information
    """
    print(f"Loading model from {checkpoint_path}...")
    
    # Initialize model with same architecture as training
    model = MBV3SmallSeg(
        num_classes=num_classes,
        backbone_pretrained=True,  # Architecture should match training
        input_size=(256, 256),
        dropout=0.1
    )
    
    # Load checkpoint (weights_only=False needed for PyTorch 2.6+)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Extract training info
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'val_miou': checkpoint.get('val_miou', 'unknown'),
        'val_loss': checkpoint.get('val_loss', 'unknown')
    }
    
    print(f"✓ Model loaded successfully!")
    print(f"  Trained for {checkpoint_info['epoch']} epochs")
    if checkpoint_info['val_miou'] != 'unknown':
        print(f"  Best validation mIoU: {checkpoint_info['val_miou']:.4f}")
        print(f"  Best validation loss: {checkpoint_info['val_loss']:.4f}")
    else:
        print(f"  Best validation mIoU: {checkpoint_info['val_miou']}")
        print(f"  Best validation loss: {checkpoint_info['val_loss']}")

    return model, checkpoint_info


def evaluate_model(model, dataloader, device, num_classes=21):
    """
    Evaluate model on dataset and compute detailed metrics.
    
    Args:
        model: Trained model in eval mode
        dataloader: DataLoader for evaluation
        device: Device for computation
        num_classes: Number of classes
    
    Returns:
        results: Dictionary with evaluation metrics
    """
    print(f"\nEvaluating model on {len(dataloader.dataset)} samples...")
    
    model.eval()
    iou_metric = IoUMetric(num_classes=num_classes, ignore_index=255)
    total_loss = 0.0
    num_batches = len(dataloader)
    
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            # Forward pass
            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += loss.item()
            
            # Get predictions
            preds = torch.argmax(logits, dim=1)
            iou_metric.update(preds, masks)
            
            # Progress
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == num_batches:
                print(f"  Processed {batch_idx + 1}/{num_batches} batches")
    
    # Compute final metrics
    avg_loss = total_loss / num_batches
    iou_results = iou_metric.compute()
    
    results = {
        'loss': avg_loss,
        'miou': iou_results['miou'],
        'iou_per_class': iou_results['iou_per_class']
    }
    
    return results


def print_detailed_results(results, class_names=VOC_CLASSES):
    """Print detailed evaluation results."""
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Average Loss: {results['loss']:.4f}")
    print(f"Mean IoU (mIoU): {results['miou']:.4f} ({results['miou']*100:.2f}%)")
    
    print(f"\nPer-class IoU:")
    print(f"{'Class':<15} {'IoU':<8} {'Percentage'}")
    print("-" * 35)
    
    iou_per_class = results['iou_per_class']
    valid_classes = []
    
    for class_id, iou in enumerate(iou_per_class):
        if iou > 0:  # Only show classes that appeared in the dataset
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
            print(f"{class_name:<15} {iou:<8.4f} {iou*100:5.1f}%")
            valid_classes.append((class_id, class_name, iou))
    
    if valid_classes:
        # Show best and worst performing classes
        best_class = max(valid_classes, key=lambda x: x[2])
        worst_class = min(valid_classes, key=lambda x: x[2])
        
        print(f"\nBest performing class: {best_class[1]} (IoU: {best_class[2]:.4f})")
        print(f"Worst performing class: {worst_class[1]} (IoU: {worst_class[2]:.4f})")
    
    print(f"{'='*60}")


def visualize_predictions(model, dataloader, device, num_samples=6, save_path='test_predictions.png'):
    """
    Visualize model predictions on sample images.
    
    Args:
        model: Trained model
        dataloader: DataLoader for getting samples
        device: Device for computation
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization
    """
    model.eval()
    
    # Get sample images
    images_list = []
    masks_list = []
    preds_list = []
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            
            # Move to CPU for visualization
            images = images.cpu()
            masks = masks.cpu()
            preds = preds.cpu()
            
            for i in range(min(images.shape[0], num_samples - len(images_list))):
                images_list.append(images[i])
                masks_list.append(masks[i])
                preds_list.append(preds[i])
                
                if len(images_list) >= num_samples:
                    break
            
            if len(images_list) >= num_samples:
                break
    
    # Create visualization
    fig, axes = plt.subplots(3, num_samples, figsize=(3*num_samples, 9))
    if num_samples == 1:
        axes = axes.reshape(3, 1)
    
    # ImageNet normalization values for denormalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for i in range(num_samples):
        if i < len(images_list):
            # Denormalize image
            img = images_list[i] * std + mean
            img = torch.clamp(img, 0, 1)
            
            # Show original image
            axes[0, i].imshow(img.permute(1, 2, 0).numpy())
            axes[0, i].set_title(f'Image {i+1}')
            axes[0, i].axis('off')
            
            # Show ground truth
            axes[1, i].imshow(masks_list[i].numpy(), cmap='tab20', vmin=0, vmax=20)
            axes[1, i].set_title('Ground Truth')
            axes[1, i].axis('off')
            
            # Show prediction
            axes[2, i].imshow(preds_list[i].numpy(), cmap='tab20', vmin=0, vmax=20)
            axes[2, i].set_title('Prediction')
            axes[2, i].axis('off')
        else:
            # Hide empty subplots
            for row in range(3):
                axes[row, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Predictions visualization saved to {save_path}")
    
    return fig


def save_results_summary(results, checkpoint_info, save_path='test_results.txt'):
    """Save evaluation results to text file."""
    with open(save_path, 'w') as f:
        f.write("MBV3SmallSeg Model Evaluation Results\n")
        f.write("="*50 + "\n\n")
        
        f.write("Model Information:\n")
        f.write(f"  Checkpoint epoch: {checkpoint_info['epoch']}\n")
        if checkpoint_info['val_miou'] != 'unknown':
            f.write(f"  Training val mIoU: {checkpoint_info['val_miou']:.4f}\n")
            f.write(f"  Training val loss: {checkpoint_info['val_loss']:.4f}\n\n")
        else:
            f.write(f"  Training val mIoU: {checkpoint_info['val_miou']}\n")
            f.write(f"  Training val loss: {checkpoint_info['val_loss']}\n\n")
        
        f.write("Validation Results:\n")
        f.write(f"  Val loss: {results['loss']:.4f}\n")
        f.write(f"  Val mIoU: {results['miou']:.4f} ({results['miou']*100:.2f}%)\n\n")
        
        f.write("Per-class IoU:\n")
        for class_id, iou in enumerate(results['iou_per_class']):
            if iou > 0:
                class_name = VOC_CLASSES[class_id] if class_id < len(VOC_CLASSES) else f"Class_{class_id}"
                f.write(f"  {class_name}: {iou:.4f}\n")
    
    print(f"Results summary saved to {save_path}")


def main():
    """Main testing function."""
    print("MBV3SmallSeg Model Testing")
    print("="*30)
    
    # Configuration
    CHECKPOINT_PATH = 'best_model.pth'
    NUM_CLASSES = 21
    IMG_SIZE = (256, 256)
    BATCH_SIZE = 16
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if checkpoint exists
    if not Path(CHECKPOINT_PATH).exists():
        print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
        print("Please make sure you have trained the model first using training_loop.py")
        return
    
    # Load model
    try:
        model, checkpoint_info = load_trained_model(CHECKPOINT_PATH, NUM_CLASSES, device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load validation data (VOC test set has no ground truth labels)
    print(f"\nLoading validation dataset...")
    try:
        val_loader = get_voc_dataloader(
            root='./archive',
            image_set='val',
            img_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4 if device.type == 'cuda' else 0,
            pin_memory=(device.type == 'cuda')
        )
        print(f"✓ Loaded {len(val_loader.dataset)} validation samples")
    except Exception as e:
        print(f"❌ Error loading validation data: {e}")
        return
    
    # Evaluate model
    start_time = time.time()
    try:
        results = evaluate_model(model, val_loader, device, NUM_CLASSES)
        eval_time = time.time() - start_time
        print(f"✓ Evaluation completed in {eval_time:.1f} seconds")
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return
    
    # Print results
    print_detailed_results(results)
    
    # Save results
    save_results_summary(results, checkpoint_info)
    
    # Visualize predictions
    print(f"\nGenerating prediction visualizations...")
    try:
        fig = visualize_predictions(model, val_loader, device, num_samples=6)
        plt.close(fig)  # Close to prevent display issues
    except Exception as e:
        print(f"⚠️  Warning: Could not generate visualizations: {e}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Model: MBV3SmallSeg")
    print(f"Dataset: VOC 2012 validation set ({len(val_loader.dataset)} images)")
    print(f"Final mIoU: {results['miou']:.4f} ({results['miou']*100:.2f}%)")
    print(f"Evaluation time: {eval_time:.1f} seconds")
    print(f"Results saved to: test_results.txt")
    print(f"Visualizations saved to: test_predictions.png")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()