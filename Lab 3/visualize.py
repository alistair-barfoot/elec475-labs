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


if __name__ == "__main__":
    main()