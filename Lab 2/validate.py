import torch
import os
from PIL import Image
import torchvision.transforms as transforms
from model import snoutNet
import cv2
import numpy as np
import argparse
import pandas as pd


def load_and_preprocess_image(image_path):
    """
    Load and preprocess an image for snoutNet inference.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        torch.Tensor: Preprocessed image tensor of shape [1, 3, 227, 227]
    """
    # Load image using PIL
    image = Image.open(image_path).convert("RGB")
    
    # Define the same transformation as used in the dataset
    transform = transforms.Compose([
        transforms.Resize((227, 227)),  # Resize to 227x227
        transforms.ToTensor()           # Convert to tensor [3, 227, 227]
    ])
    
    # Apply transformation and add batch dimension
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension: [1, 3, 227, 227]
    
    return image_tensor


def load_model(model_path):
    """
    Load the trained snoutNet model from file.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        torch.nn.Module: Loaded model in evaluation mode
    """
    model = snoutNet()
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model


def predict_nose_coordinates(image_path, model_path="models/snoutnet_weights.pth"):
    """
    Predict nose coordinates for a given image.
    
    Args:
        image_path (str): Path to the input image
        model_path (str): Path to the trained model
        
    Returns:
        tuple: (x_coord, y_coord) in original image scale
    """
    image_tensor = load_and_preprocess_image(image_path)
    model = load_model(model_path)
    
    with torch.no_grad():
        output = model(image_tensor)
    
    # Scale coordinates from 227x227 to original image size
    img = cv2.imread(image_path)
    x_coord = img.shape[1] * output[0, 0].item() / 227
    y_coord = img.shape[0] * output[0, 1].item() / 227
    
    return x_coord, y_coord


def parse_ground_truth_coordinates(coord_string):
    """
    Parse ground truth coordinates from string format.
    
    Args:
        coord_string (str): Coordinate string in format "(x, y)"
        
    Returns:
        tuple: (x, y) as integers
    """
    x_str, y_str = coord_string.strip('()').split(', ')
    return int(x_str), int(y_str)


def calculate_statistics(distances):
    """
    Calculate mean and standard deviation of distances.
    
    Args:
        distances (list): List of distance values
        
    Returns:
        tuple: (mean, std_dev)
    """
    return np.mean(distances), np.std(distances)


def visualize_prediction(image_path, predicted_coords, ground_truth_coords=None, show_ground_truth=False):
    """
    Visualize the predicted nose position on the image.
    
    Args:
        image_path (str): Path to the image
        predicted_coords (tuple): Predicted (x, y) coordinates
        ground_truth_coords (tuple, optional): Ground truth (x, y) coordinates
        show_ground_truth (bool): Whether to show ground truth coordinates
    """
    img = cv2.imread(image_path)
    
    # Draw predicted nose (green circle)
    cv2.circle(img, (int(predicted_coords[0]), int(predicted_coords[1])), 5, (0, 255, 0), -1)
    
    # Draw ground truth nose (blue circle) if requested
    if show_ground_truth and ground_truth_coords:
        cv2.circle(img, ground_truth_coords, 5, (255, 0, 0), -1)
    
    cv2.imshow("Predicted Nose Position", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description='Validate snoutNet model on test images')
    parser.add_argument('-g', '--ground', action='store_true', 
                       help='Show ground truth coordinates')
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='Print detailed results for each image')
    parser.add_argument('-s', '--show', action='store_true', 
                       help='Display images with predicted coordinates')
    args = parser.parse_args()

    # Load ground truth data
    labels = pd.read_csv('test_noses.txt', header=None, names=['filename', 'coordinates'])
    distances = []

    print("Validating model predictions...")
    
    for _, row in labels.iterrows():
        filename = row['filename']
        image_path = os.path.join('images', filename)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Get ground truth coordinates
        ground_x, ground_y = parse_ground_truth_coordinates(row['coordinates'])
        
        # Get predicted coordinates
        pred_x, pred_y = predict_nose_coordinates(image_path)
        
        # Calculate distance
        distance = np.linalg.norm(np.array([pred_x, pred_y]) - np.array([ground_x, ground_y]))
        distances.append(distance)
        
        if args.verbose:
            print(f"Image: {filename}")
            print(f"Ground truth: ({ground_x}, {ground_y})")
            print(f"Predicted: ({pred_x:.2f}, {pred_y:.2f})")
            print(f"Distance: {distance:.2f} pixels")
            print("-" * 50)
        
        if args.show:
            visualize_prediction(image_path, (pred_x, pred_y), 
                               (ground_x, ground_y), args.ground)
    
    # Calculate and display statistics
    if distances:
        mean, std_dev = calculate_statistics(distances)
        print(f"\nValidation Results:")
        print(f"Count: {len(distances)}")
        print(f"Mean: {mean:.1f} pixels")
        print(f"Std Dev: {std_dev:.1f} pixels")
        print(f"Min: {min(distances):.1f} pixels")
        print(f"Max: {max(distances):.1f} pixels")
    else:
        print("No valid images found for validation.")


if __name__ == "__main__":
    main()