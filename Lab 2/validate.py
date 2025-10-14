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
        return {
            'count': len(distances),
            'mean': mean,
            'std_dev': std_dev,
            'min': min(distances),
            'max': max(distances)
        }
    else:
        return None


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
        
        results_summary = []
        
        for model_path in sorted(model_files):
            model_name = os.path.basename(model_path)
            print(f"Validating model: {model_name}")
            
            try:
                results = validate_single_model(model_path, labels, args)
                
                if results:
                    print(f"Results for {model_name}:")
                    print(f"  Count: {results['count']}")
                    print(f"  Mean: {results['mean']:.1f} pixels")
                    print(f"  Std Dev: {results['std_dev']:.1f} pixels")
                    print(f"  Min: {results['min']:.1f} pixels")
                    print(f"  Max: {results['max']:.1f} pixels")
                    
                    results_summary.append({
                        'model': model_name,
                        'mean': results['mean'],
                        'std_dev': results['std_dev'],
                        'count': results['count'],
                        'min': results['min'],
                        'max': results['max']
                    })
                else:
                    print(f"  No valid results for {model_name}")
                    
            except Exception as e:
                print(f"  Error validating {model_name}: {e}")
            
            print("-" * 60)
        
        # Print summary comparison
        if results_summary:
            print("\nSUMMARY COMPARISON:")
            print("=" * 80)
            print(f"{'Model':<30} {'Mean':<10} {'Std Dev':<10} {'Min':<10} {'Max':<10} ")
            print("-" * 80)
            
            # Sort by mean error (best first)
            results_summary.sort(key=lambda x: x['mean'])
            
            for result in results_summary:
                print(f"{result['model']:<30} {result['mean']:<10.1f} {result['std_dev']:<10.1f} {result['min']:<10.1f} {result['max']:<10.1f}")
            print(f"\nBest performing model: {results_summary[0]['model']} (Mean: {results_summary[0]['mean']:.1f} pixels)")
    
    else:
        # Single model validation (original behavior)
        model_path = args.model
        
        print(f"Validating single model: {os.path.basename(model_path)}")
        
        results = validate_single_model(model_path, labels, args)
        
        if results:
            print(f"\nValidation Results:")
            print(f"Count: {results['count']}")
            print(f"Mean: {results['mean']:.1f} pixels")
            print(f"Std Dev: {results['std_dev']:.1f} pixels")
            print(f"Min: {results['min']:.1f} pixels")
            print(f"Max: {results['max']:.1f} pixels")
        else:
            print("No valid images found for validation.")


if __name__ == "__main__":
    main()