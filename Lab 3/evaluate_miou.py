"""
Standalone mIoU Evaluation Script for FCN ResNet50
==================================================

This script calculates mean Intersection over Union (mIoU) for semantic segmentation.
"""

import cv2
import numpy as np
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
import torch
from PIL import Image
import os
import random

def load_image(image_id):
    """Load image from dataset"""
    img_path = f'archive/VOC2012_train_val/VOC2012_train_val/JPEGImages/{image_id}.jpg'
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def load_ground_truth(image_id):
    """Load ground truth segmentation mask"""
    mask_path = f'archive/VOC2012_train_val/VOC2012_train_val/SegmentationClass/{image_id}.png'
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    return mask

def predict_segmentation(image, model, preprocess):
    """Get segmentation prediction from FCN model"""
    # Convert to PIL and preprocess
    img_pil = Image.fromarray(image)
    img_tensor = preprocess(img_pil).unsqueeze(0)
    
    # Get prediction
    with torch.no_grad():
        output = model(img_tensor)
    
    # Process output
    pred_logits = output['out'][0]
    pred_mask = torch.argmax(pred_logits, dim=0).cpu().numpy()
    
    # Resize to original image size
    original_height, original_width = image.shape[:2]
    pred_mask = cv2.resize(
        pred_mask.astype(np.uint8),
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST
    )
    
    return pred_mask

def calculate_iou_single_class(pred_mask, gt_mask, class_id):
    """Calculate IoU for a single class"""
    pred_class = (pred_mask == class_id)
    gt_class = (gt_mask == class_id)
    
    intersection = np.logical_and(pred_class, gt_class).sum()
    union = np.logical_or(pred_class, gt_class).sum()
    
    if union == 0:
        return None  # Class not present
    
    return intersection / union

def calculate_miou(pred_mask, gt_mask, num_classes=21):
    """Calculate mean IoU"""
    # Handle PASCAL VOC ignore label (255)
    gt_mask = np.where(gt_mask == 255, 0, gt_mask)
    
    ious = []
    valid_classes = []
    
    for class_id in range(num_classes):
        iou = calculate_iou_single_class(pred_mask, gt_mask, class_id)
        if iou is not None:
            ious.append(iou)
            valid_classes.append(class_id)
    
    miou = np.mean(ious) if ious else 0.0
    return miou, ious, valid_classes

def get_class_names():
    """Get PASCAL VOC class names"""
    return [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
        'sofa', 'train', 'tvmonitor'
    ]

def evaluate_single_image(image_id, model, preprocess, verbose=True):
    """Evaluate mIoU on a single image"""
    try:
        # Load data
        image = load_image(image_id)
        gt_mask = load_ground_truth(image_id)
        
        # Get prediction
        pred_mask = predict_segmentation(image, model, preprocess)
        
        # Calculate mIoU
        miou, ious, valid_classes = calculate_miou(pred_mask, gt_mask)
        
        if verbose:
            class_names = get_class_names()
            print(f"\nImage: {image_id}")
            print(f"mIoU: {miou:.4f} ({miou*100:.1f}%)")
            print(f"Classes present: {len(valid_classes)}")
            
            # Show per-class results
            for class_id, iou in zip(valid_classes, ious):
                class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
                print(f"  {class_name:12s}: {iou:.3f}")
        
        return miou, len(valid_classes)
        
    except Exception as e:
        if verbose:
            print(f"Error processing {image_id}: {e}")
        return None, 0

def evaluate_dataset(num_images=10, use_val_set=True):
    """Evaluate mIoU on multiple images"""
    print("Loading FCN ResNet50 model...")
    weights = FCN_ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    model = fcn_resnet50(weights=weights, progress=False)
    model.eval()
    print("Model loaded!")
    
    # Get image IDs
    if use_val_set:
        ids_file = 'archive/VOC2012_train_val/VOC2012_train_val/ImageSets/Segmentation/val.txt'
    else:
        ids_file = 'archive/VOC2012_train_val/VOC2012_train_val/ImageSets/Segmentation/train.txt'
    
    try:
        with open(ids_file, 'r') as f:
            image_ids = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"Could not find {ids_file}")
        return
    
    # Sample random images
    if len(image_ids) > num_images:
        image_ids = random.sample(image_ids, num_images)
    
    print(f"\nEvaluating {len(image_ids)} images...")
    print("=" * 50)
    
    all_mious = []
    total_classes = 0
    
    for i, image_id in enumerate(image_ids):
        print(f"\n[{i+1}/{len(image_ids)}] Processing {image_id}...")
        
        miou, num_classes = evaluate_single_image(image_id, model, preprocess, verbose=False)
        
        if miou is not None:
            all_mious.append(miou)
            total_classes += num_classes
            print(f"  mIoU: {miou:.4f} ({num_classes} classes)")
        else:
            print(f"  Failed to process")
    
    # Final results
    if all_mious:
        avg_miou = np.mean(all_mious)
        std_miou = np.std(all_mious)
        
        print("\n" + "=" * 50)
        print("FINAL RESULTS")
        print("=" * 50)
        print(f"Images processed: {len(all_mious)}/{len(image_ids)}")
        print(f"Average mIoU: {avg_miou:.4f} ± {std_miou:.4f}")
        print(f"Average mIoU (%): {avg_miou*100:.2f}% ± {std_miou*100:.2f}%")
        print(f"Best mIoU: {max(all_mious):.4f} ({max(all_mious)*100:.1f}%)")
        print(f"Worst mIoU: {min(all_mious):.4f} ({min(all_mious)*100:.1f}%)")
        print(f"Average classes per image: {total_classes/len(all_mious):.1f}")
        print("=" * 50)
        
        return avg_miou
    else:
        print("No images were successfully processed!")
        return 0.0

def main():
    """Main evaluation function"""
    print("FCN ResNet50 mIoU Evaluation")
    print("=" * 30)
    
    # Quick single image test
    print("\n1. Single Image Test:")
    weights = FCN_ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    model = fcn_resnet50(weights=weights, progress=False)
    model.eval()
    
    # Test on a random image
    train_file = 'archive/VOC2012_train_val/VOC2012_train_val/ImageSets/Segmentation/train.txt'
    with open(train_file, 'r') as f:
        train_ids = [line.strip() for line in f.readlines()]
    
    test_image = random.choice(train_ids)
    miou, num_classes = evaluate_single_image(test_image, model, preprocess, verbose=True)
    
    # Multiple image evaluation
    print(f"\n2. Multiple Image Evaluation:")
    choice = input(f"Evaluate on multiple images? (y/n): ").lower()
    
    if choice == 'y':
        num_images = input("Number of images to evaluate (default 10): ")
        try:
            num_images = int(num_images) if num_images else 10
        except:
            num_images = 10
        
        avg_miou = evaluate_dataset(num_images=num_images, use_val_set=True)

if __name__ == "__main__":
    main()