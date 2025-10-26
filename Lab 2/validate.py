import torch
import os
from PIL import Image
import torchvision.transforms as transforms
from model import snoutNet
import cv2
import numpy as np
import argparse
import pandas as pd
import glob


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
    
    # Resize image and convert to tensor
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
    x_coord, y_coord = int(x_coord), int(y_coord)
    
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
        ground_truth_coords = (int(ground_truth_coords[0]), int(ground_truth_coords[1]))
        cv2.circle(img, ground_truth_coords, 5, (255, 0, 0), -1)
    
    cv2.imshow("Predicted Nose Position", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def validate_single_model(model_path, labels, args):
    """
    Validate a single model and return results.
    
    Args:
        model_path (str): Path to the model file
        labels (pd.DataFrame): Ground truth data
        args: Command line arguments
        
    Returns:
        dict: Validation results
    """
    distances = []
    
    for _, row in labels.iterrows():
        filename = row['filename']
        image_path = os.path.join('images', filename)
        img = cv2.imread(image_path)
        
        if not os.path.exists(image_path):
            if args.verbose:
                print(f"Warning: Image not found: {image_path}")
            continue
        
        try:
            # Get ground truth coordinates
            ground_x, ground_y = parse_ground_truth_coordinates(row['coordinates'])
            ground_x /= img.shape[1] / 227
            ground_y /= img.shape[0] / 227

            # Get predicted coordinates
            pred_x, pred_y = predict_nose_coordinates(image_path, model_path=model_path)
            pred_x /= img.shape[1] / 227
            pred_y /= img.shape[0] / 227

            # Calculate distance
            distance = np.linalg.norm(np.array([pred_x, pred_y]) - np.array([ground_x, ground_y]))
            distances.append(distance)

            pred_x *= img.shape[1] / 227
            pred_y *= img.shape[0] / 227

            ground_x *= img.shape[1] / 227
            ground_y *= img.shape[0] / 227

            if args.verbose and not args.all:
                print(f"Image: {filename}")
                print(f"Ground truth: ({ground_x}, {ground_y})")
                print(f"Predicted: ({pred_x:.2f}, {pred_y:.2f})")
                print(f"Relative Distance: {distance:.2f} pixels")
                print("-" * 50)
            
            if args.show and not args.all:
                visualize_prediction(image_path, (pred_x, pred_y), 
                                   (ground_x, ground_y), args.ground)
        
        except Exception as e:
            if args.verbose:
                print(f"Error processing {filename}: {e}")
            continue
    
    # Calculate statistics
    if distances:
        mean, std_dev = calculate_statistics(distances)
        # Find mean, std for 4 worst
        worst_distances = sorted(distances)[-4:]
        worst_mean, worst_std_dev = calculate_statistics(worst_distances)
        worst_min, worst_max = min(worst_distances), max(worst_distances)

        # Find mean, std for 4 best
        best_distances = sorted(distances)[:4]
        best_mean, best_std_dev = calculate_statistics(best_distances)
        best_min, best_max = min(best_distances), max(best_distances)
        return {
            'count': len(distances),
            'mean': mean,
            'std_dev': std_dev,
            'min': min(distances),
            'max': max(distances),
            'worst_mean': worst_mean,
            'worst_std_dev': worst_std_dev,
            'worst_min': worst_min,
            'worst_max': worst_max,
            'best_mean': best_mean,
            'best_std_dev': best_std_dev,
            'best_min': best_min,
            'best_max': best_max
        }
    else:
        return None

def report_statistics(modelname, results):
    print("-"*60)
    print(f"Results for {modelname}:")
    print(f"  Overall - Min: {results['min']:3.0f}, Max: {results['max']:3.0f}, Mean: {results['mean']:3.0f}, Stdev: {results['std_dev']:3.1f}")
    print(f"  Worst 4 - Min: {results['worst_min']:3.0f}, Max: {results['worst_max']:3.0f}, Mean: {results['worst_mean']:3.0f}, Stdev: {results['worst_std_dev']:3.1f}")
    print(f"  Best 4  - Min: {results['best_min']:3.0f}, Max: {results['best_max']:3.0f}, Mean: {results['best_mean']:3.0f}, Stdev: {results['best_std_dev']:3.1f}")
    print("-"*60)
    

def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description='Validate snoutNet model on test images')
    parser.add_argument('-g', '--ground', action='store_true', 
               help='Show ground truth coordinates')
    parser.add_argument('-v', '--verbose', action='store_true', 
               help='Print detailed results for each image')
    parser.add_argument('-s', '--show', action='store_true', 
               help='Display images with predicted coordinates')
    parser.add_argument('-m', '--model', type=str, default='models/snoutnet_weights.pth',
               help='Path to the trained model file')
    parser.add_argument('-a', '--all', action='store_true',
               help='Validate all .pth files in the models folder')
    args = parser.parse_args()

    # Load ground truth data
    labels = pd.read_csv('test_noses.txt', header=None, names=['filename', 'coordinates'])

    if args.all:
      # Get all .pth files in the models folder
      model_files = glob.glob('models/*.pth')
      if not model_files:
        print("No .pth files found in the models folder.")
        return
      print(f"Found {len(model_files)} model files. Validating all models...\n")
    else:
      model_files = [args.model]

    for model_path in sorted(model_files):
      model_name = os.path.basename(model_path)
      print(f"Validating model: {model_name}")
      
      results = validate_single_model(model_path, labels, args)
      
      if results:
        report_statistics(os.path.basename(model_path), results)
      else:
        print("No valid images found for validation.")


if __name__ == "__main__":
    main()