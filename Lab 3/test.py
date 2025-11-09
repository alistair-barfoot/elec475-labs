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
import argparse
import random

from MobileNet_model import MBV3SmallSeg
from dataloader import get_voc_dataloader, VOC_CLASSES
from training_loop import IoUMetric


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Test MBV3SmallSeg model on VOC 2012 validation set',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test.py -r              # Run evaluation and save results only
  python test.py -v              # Generate visualizations only
  python test.py -r -v           # Run both evaluation and visualization
  python test.py                 # Run both (default behavior)
        """
    )
    
    parser.add_argument('-r', '--results', 
                        action='store_true',
                        help='Run model evaluation and save results')
    
    parser.add_argument('-v', '--visualize',
                        action='store_true', 
                        help='Generate prediction visualizations')
    
    parser.add_argument('--checkpoint',
                        type=str,
                        default='best_model.pth',
                        help='Path to model checkpoint (default: best_model.pth)')
    
    parser.add_argument('--batch-size',
                        type=int,
                        default=16,
                        help='Batch size for evaluation (default: 16)')
    
    parser.add_argument('--num-samples',
                        type=int,
                        default=6,
                        help='Number of samples to visualize (default: 6)')
    
    args = parser.parse_args()
    
    # If no flags specified, run both by default
    if not args.results and not args.visualize:
        args.results = True
        args.visualize = True
    
    return args


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
    
    # Handle different checkpoint formats (regular training vs knowledge distillation)
    if 'student_state_dict' in checkpoint:
        # Knowledge distillation checkpoint
        model.load_state_dict(checkpoint['student_state_dict'])
        val_loss_key = 'val_total_loss'
    elif 'model_state_dict' in checkpoint:
        # Regular training checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        val_loss_key = 'val_loss'
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)
        val_loss_key = None
    
    model = model.to(device)
    model.eval()
    
    # Extract training info
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'val_miou': checkpoint.get('val_miou', 'unknown'),
        'val_loss': checkpoint.get(val_loss_key, 'unknown') if val_loss_key else 'unknown'
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
  
  # Collect all samples from the dataset
  all_images = []
  all_masks = []
  all_preds = []
  
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
      
      for i in range(images.shape[0]):
        all_images.append(images[i])
        all_masks.append(masks[i])
        all_preds.append(preds[i])
  
  # Randomly select samples
  total_samples = len(all_images)
  if total_samples < num_samples:
    print(f"Warning: Only {total_samples} samples available, showing all of them")
    num_samples = total_samples
  
  random_indices = random.sample(range(total_samples), num_samples)
  
  # Get selected samples
  images_list = [all_images[i] for i in random_indices]
  masks_list = [all_masks[i] for i in random_indices]
  preds_list = [all_preds[i] for i in random_indices]
  
  # Create visualization
  fig, axes = plt.subplots(3, num_samples, figsize=(3*num_samples, 9))
  if num_samples == 1:
    axes = axes.reshape(3, 1)
  
  # ImageNet normalization values for denormalization
  mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
  std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
  
  for i in range(num_samples):
    # Denormalize image
    img = images_list[i] * std + mean
    img = torch.clamp(img, 0, 1)
    
    # Show original image
    axes[0, i].imshow(img.permute(1, 2, 0).numpy())
    axes[0, i].set_title(f'Image {random_indices[i]+1}')
    axes[0, i].axis('off')
    
    # Show ground truth
    axes[1, i].imshow(masks_list[i].numpy(), cmap='tab20', vmin=0, vmax=20)
    axes[1, i].set_title('Ground Truth')
    axes[1, i].axis('off')
    
    # Calculate mIoU for this individual prediction
    pred_mask = preds_list[i].numpy()
    gt_mask = masks_list[i].numpy()
    
    # Calculate IoU for each class present
    valid_ious = []
    for class_id in range(21):  # VOC has 21 classes (0-20)
      # Skip ignore pixels (255) and classes not present in ground truth
      if class_id == 255 or not np.any(gt_mask == class_id):
        continue
      
      # Calculate intersection and union for this class
      pred_class = (pred_mask == class_id)
      gt_class = (gt_mask == class_id)
      
      intersection = np.logical_and(pred_class, gt_class).sum()
      union = np.logical_or(pred_class, gt_class).sum()
      
      if union > 0:
        iou = intersection / union
        valid_ious.append(iou)
    
    # Calculate mean IoU for this sample
    sample_miou = np.mean(valid_ious) if valid_ious else 0.0
    
    # Show prediction with mIoU
    axes[2, i].imshow(pred_mask, cmap='tab20', vmin=0, vmax=20)
    axes[2, i].set_title(f'Prediction\nmIoU: {sample_miou:.3f}')
    axes[2, i].axis('off')
  
  plt.tight_layout()
  
  # Add legend at the bottom
  # Create a separate subplot for the legend
  fig.subplots_adjust(bottom=0.15)  # Make room for legend
  
  # Create colormap and legend
  import matplotlib.patches as mpatches
  from matplotlib.colors import ListedColormap
  import matplotlib.cm as cm
  
  # Get the tab20 colormap
  tab20 = cm.get_cmap('tab20')
  colors = [tab20(i) for i in range(21)]  # 21 VOC classes (0-20)
  
  # Create legend patches
  legend_patches = []
  for i, class_name in enumerate(VOC_CLASSES):
    if i < len(colors):
      patch = mpatches.Patch(color=colors[i], label=f'{i}: {class_name}')
      legend_patches.append(patch)
  
  # Add legend below the plots
  fig.legend(handles=legend_patches, 
             loc='lower center', 
             ncol=7,  # 7 columns for better layout
             bbox_to_anchor=(0.5, 0.02),
             fontsize=8,
             frameon=True,
             fancybox=True,
             shadow=True)
  
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
    # Parse command line arguments
    args = parse_arguments()
    
    print("MBV3SmallSeg Model Testing")
    print("="*30)
    print(f"Running: {'Results ' if args.results else ''}{'Visualizations' if args.visualize else ''}")
    print()
    
    # Configuration
    CHECKPOINT_PATH = args.checkpoint
    NUM_CLASSES = 21
    IMG_SIZE = (256, 256)
    BATCH_SIZE = args.batch_size
    
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
    
    # Initialize results variable
    results = None
    eval_time = 0
    
    # Run evaluation if requested
    if args.results:
        start_time = time.time()
        try:
            results = evaluate_model(model, val_loader, device, NUM_CLASSES)
            eval_time = time.time() - start_time
            print(f"✓ Evaluation completed in {eval_time:.1f} seconds")
            
            # Print and save results
            print_detailed_results(results)
            save_results_summary(results, checkpoint_info)
            
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            return
    
    # Generate visualizations if requested
    if args.visualize:
        print(f"\nGenerating prediction visualizations...")
        try:
            fig = visualize_predictions(model, val_loader, device, num_samples=args.num_samples)
            plt.close(fig)  # Close to prevent display issues
        except Exception as e:
            print(f"⚠️  Warning: Could not generate visualizations: {e}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Model: MBV3SmallSeg")
    print(f"Dataset: VOC 2012 validation set ({len(val_loader.dataset)} images)")
    
    if args.results and results:
        print(f"Final mIoU: {results['miou']:.4f} ({results['miou']*100:.2f}%)")
        print(f"Evaluation time: {eval_time:.1f} seconds")
        print(f"Results saved to: test_results.txt")
    
    if args.visualize:
        print(f"Visualizations saved to: test_predictions.png")
    
    print(f"{'='*60}")


if __name__ == '__main__':
    main()