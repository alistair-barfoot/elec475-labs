import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
import torch
import torchvision.transforms as transforms
from PIL import Image
import random

def main():
    print("Loading FCN ResNet50 model...")
    weights = FCN_ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()

    model = fcn_resnet50(weights=weights, progress=True)
    model.eval()
    print("Model loaded successfully!")

    # Load training images 
    train_dir = 'archive/VOC2012_train_val/VOC2012_train_val/ImageSets/Segmentation/train.txt'
    train_ids = []

    with open(train_dir, 'r') as f:
        train_ids = [line.strip() for line in f.readlines()]

    # Test with an image from your dataset (using correct image ID format)
    image_id = random.choice(train_ids)

    print(f"\nProcessing image: {image_id}")
    
    img_rgb = load_image(image_id)
    print(f"Image loaded: {img_rgb.shape}")

    mask_rgb = load_mask(image_id)
    print(f"Ground truth mask loaded: {mask_rgb.shape}")
    
    output_mask = test_model(image_id, model, preprocess)
    print(f"Segmentation complete: {output_mask.shape}")
    
    # Print statistics about the segmentation
    print_mask_statistics(output_mask)
    
    # Calculate and print mIoU
    miou = print_miou_results(output_mask, mask_rgb)
    
    # Display results
    show_mask_on_image(img_rgb, mask_rgb, output_mask)

def load_image(image_id):
    """Load image from your dataset"""
    img_path = f'archive/VOC2012_train_val/VOC2012_train_val/JPEGImages/{image_id}.jpg'
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def load_mask(image_id): 
    """Load ground truth mask from your dataset"""
    mask_path = f'archive/VOC2012_train_val/VOC2012_train_val/SegmentationClass/{image_id}.png'
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    return mask

def test_model(image_id, model, preprocess):
    """Run image through FCN model and return processed mask"""
    # Load image
    img_rgb = load_image(image_id)
    original_height, original_width = img_rgb.shape[:2]
    
    # Convert numpy array to PIL Image for preprocessing
    img_pil = Image.fromarray(img_rgb)
    
    # Preprocess image for the model
    img_tensor = preprocess(img_pil).unsqueeze(0)  # Add batch dimension

    # Forward pass
    with torch.no_grad():
        output = model(img_tensor)
    
    # Process the output
    # FCN outputs a dictionary with 'out' key containing the segmentation logits
    segmentation_logits = output['out'][0]  # Remove batch dimension
    
    # Get the predicted class for each pixel (argmax across classes)
    predicted_mask = torch.argmax(segmentation_logits, dim=0).cpu().numpy()
    
    # Resize the predicted mask back to original image size
    predicted_mask_resized = cv2.resize(
        predicted_mask.astype(np.uint8), 
        (original_width, original_height), 
        interpolation=cv2.INTER_NEAREST  # Use nearest neighbor to preserve class labels
    )
    
    print(f"Original image size: {original_height}x{original_width}")
    print(f"Model output size: {predicted_mask.shape}")
    print(f"Resized mask size: {predicted_mask_resized.shape}")
    
    return predicted_mask_resized

def show_mask_on_image(image, ground_truth, mask):
    """Display original image and predicted segmentation mask"""
    plt.figure(figsize=(15, 5))
    
    # Show original image
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Show predicted mask
    plt.subplot(1, 4, 2)
    plt.imshow(mask, cmap='tab20')  # Use a colormap that shows different classes
    plt.title('Predicted Segmentation Mask')
    plt.axis('off')

    # Show ground truth mask
    plt.subplot(1, 4, 3)
    plt.imshow(ground_truth, cmap='tab20')  # Use a colormap that shows different classes
    plt.title('Ground Truth Mask')
    plt.axis('off')
    
    # Show overlay
    plt.subplot(1, 4, 4)
    plt.imshow(image)
    plt.imshow(mask, alpha=0.5, cmap='tab20')  # Semi-transparent overlay
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('segmentation_result.png', dpi=150, bbox_inches='tight')
    print("Result saved as 'segmentation_result.png'")
    
    # Don't show interactively to avoid blocking
    # plt.show()

def calculate_iou_per_class(pred_mask, gt_mask, num_classes=21):
    """
    Calculate IoU for each class
    
    Args:
        pred_mask: Predicted segmentation mask
        gt_mask: Ground truth segmentation mask
        num_classes: Number of classes (21 for PASCAL VOC)
    
    Returns:
        iou_per_class: IoU for each class
        valid_classes: Classes that appear in ground truth
    """
    iou_per_class = []
    valid_classes = []
    
    for class_id in range(num_classes):
        # Get pixels for this class
        pred_class = (pred_mask == class_id)
        gt_class = (gt_mask == class_id)
        
        # Calculate intersection and union
        intersection = np.logical_and(pred_class, gt_class).sum()
        union = np.logical_or(pred_class, gt_class).sum()
        
        # Only calculate IoU if this class appears in ground truth
        if union > 0:
            iou = intersection / union
            iou_per_class.append(iou)
            valid_classes.append(class_id)
    
    return np.array(iou_per_class), valid_classes

def calculate_miou(pred_mask, gt_mask, num_classes=21):
    """
    Calculate mean Intersection over Union (mIoU)
    
    Args:
        pred_mask: Predicted segmentation mask
        gt_mask: Ground truth segmentation mask
        num_classes: Number of classes (21 for PASCAL VOC)
    
    Returns:
        miou: Mean IoU
        iou_per_class: IoU for each class
        class_names: Names of valid classes
    """
    # Handle potential size mismatches
    if pred_mask.shape != gt_mask.shape:
        print(f"Warning: Mask size mismatch. Pred: {pred_mask.shape}, GT: {gt_mask.shape}")
        # Resize predicted mask to match ground truth
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), 
                              (gt_mask.shape[1], gt_mask.shape[0]), 
                              interpolation=cv2.INTER_NEAREST)
    
    # PASCAL VOC uses 255 as ignore/void class, convert to background (0)
    gt_mask = np.where(gt_mask == 255, 0, gt_mask)
    
    iou_per_class, valid_classes = calculate_iou_per_class(pred_mask, gt_mask, num_classes)
    
    # Calculate mean IoU
    miou = np.mean(iou_per_class) if len(iou_per_class) > 0 else 0.0
    
    # Get class names
    classes = create_class_colormap()
    class_names = [classes[i] if i < len(classes) else f"Class_{i}" for i in valid_classes]
    
    return miou, iou_per_class, class_names, valid_classes

def print_miou_results(pred_mask, gt_mask):
    """Print detailed mIoU results"""
    miou, iou_per_class, class_names, valid_classes = calculate_miou(pred_mask, gt_mask)
    
    print("\n" + "="*60)
    print("mIoU EVALUATION RESULTS")
    print("="*60)
    print(f"Mean IoU (mIoU): {miou:.4f} ({miou*100:.2f}%)")
    print(f"Number of classes evaluated: {len(valid_classes)}")
    
    print(f"\nPer-class IoU:")
    print("-" * 40)
    for i, (class_id, class_name, iou) in enumerate(zip(valid_classes, class_names, iou_per_class)):
        print(f"Class {class_id:2d} ({class_name:12s}): {iou:.4f} ({iou*100:5.1f}%)")
    
    # Additional statistics
    best_class_idx = np.argmax(iou_per_class)
    worst_class_idx = np.argmin(iou_per_class)
    
    print(f"\nBest performing class: {class_names[best_class_idx]} (IoU: {iou_per_class[best_class_idx]:.4f})")
    print(f"Worst performing class: {class_names[worst_class_idx]} (IoU: {iou_per_class[worst_class_idx]:.4f})")
    print("="*60)
    
    return miou

def create_class_colormap():
    """Create a colormap for PASCAL VOC classes"""
    # PASCAL VOC 2012 has 21 classes (including background)
    classes = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
        'sofa', 'train', 'tvmonitor'
    ]
    return classes

def print_mask_statistics(mask):
    """Print statistics about the predicted mask"""
    classes = create_class_colormap()
    unique_classes = np.unique(mask)
    
    print("\nSegmentation Results:")
    print("====================")
    total_pixels = mask.shape[0] * mask.shape[1]
    
    for class_id in unique_classes:
        if class_id < len(classes):
            class_name = classes[class_id]
            pixel_count = np.sum(mask == class_id)
            percentage = (pixel_count / total_pixels) * 100
            print(f"Class {class_id:2d} ({class_name:12s}): {pixel_count:6d} pixels ({percentage:5.1f}%)")


def evaluate_multiple_images(model, preprocess, num_images=10):
    """
    Evaluate mIoU across multiple images
    
    Args:
        model: Trained FCN model
        preprocess: Preprocessing transforms
        num_images: Number of images to evaluate
    
    Returns:
        avg_miou: Average mIoU across all images
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING mIoU ON {num_images} IMAGES")
    print(f"{'='*60}")
    
    # Load validation image IDs
    val_dir = 'archive/VOC2012_train_val/VOC2012_train_val/ImageSets/Segmentation/val.txt'
    val_ids = []
    
    try:
        with open(val_dir, 'r') as f:
            val_ids = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        # Fallback to train IDs if val not available
        train_dir = 'archive/VOC2012_train_val/VOC2012_train_val/ImageSets/Segmentation/train.txt'
        with open(train_dir, 'r') as f:
            val_ids = [line.strip() for line in f.readlines()]
        print("Using train set for evaluation (val set not found)")
    
    # Randomly sample images
    if len(val_ids) > num_images:
        val_ids = random.sample(val_ids, num_images)
    
    all_mious = []
    all_class_ious = []
    
    for i, image_id in enumerate(val_ids):
        print(f"\nProcessing image {i+1}/{len(val_ids)}: {image_id}")
        
        try:
            # Load image and ground truth
            img_rgb = load_image(image_id)
            gt_mask = load_mask(image_id)
            
            # Get prediction
            pred_mask = test_model(image_id, model, preprocess)
            
            # Calculate mIoU for this image
            miou, iou_per_class, class_names, valid_classes = calculate_miou(pred_mask, gt_mask)
            all_mious.append(miou)
            
            print(f"  mIoU: {miou:.4f} ({len(valid_classes)} classes)")
            
        except Exception as e:
            print(f"  Error processing {image_id}: {e}")
            continue
    
    # Calculate overall statistics
    if all_mious:
        avg_miou = np.mean(all_mious)
        std_miou = np.std(all_mious)
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Images evaluated: {len(all_mious)}")
        print(f"Average mIoU: {avg_miou:.4f} ± {std_miou:.4f}")
        print(f"Best mIoU: {max(all_mious):.4f}")
        print(f"Worst mIoU: {min(all_mious):.4f}")
        print(f"{'='*60}")
        
        return avg_miou
    else:
        print("No images were successfully processed!")
        return 0.0


if __name__ == "__main__":
    main()
    
    # Uncomment the following lines to evaluate on multiple images
    # print("\n" + "="*60)
    # print("Would you like to evaluate on multiple images? (y/n)")
    # response = input().lower()
    # if response == 'y':
    #     weights = FCN_ResNet50_Weights.DEFAULT
    #     preprocess = weights.transforms()
    #     model = fcn_resnet50(weights=weights, progress=False)
    #     model.eval()
    #     evaluate_multiple_images(model, preprocess, num_images=20)