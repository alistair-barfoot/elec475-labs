#!/usr/bin/env python3
"""
Simple script to run the nose detection training with the correct parameters.
This demonstrates how to use the updated training script.
"""

import subprocess
import sys
import os

def main():
    """Run the training script with appropriate parameters"""
    
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, "images")
    train_labels = os.path.join(script_dir, "train_noses.txt")
    test_labels = os.path.join(script_dir, "test_noses.txt")
    
    # Check if required files exist
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        return 1
    
    if not os.path.exists(train_labels):
        print(f"Error: Training labels file not found: {train_labels}")
        return 1
    
    if not os.path.exists(test_labels):
        print(f"Error: Test labels file not found: {test_labels}")
        return 1
    
    # Build command
    cmd = [
        sys.executable, "train.py",
        "--images_dir", images_dir,
        "--train_labels", train_labels,
        "--test_labels", test_labels,
        "--batch_size", "16",  # Smaller batch size for better stability
        "--epochs", "25",      # Fewer epochs for initial testing
        "--learning_rate", "0.0001",  # Lower learning rate for regression
        "--val_split", "0.2",
        "--save_path", "snoutNet_nose_detection.pth"
    ]
    
    print("Starting nose detection training...")
    print("Command:", " ".join(cmd))
    print("-" * 50)
    
    # Run the training
    try:
        result = subprocess.run(cmd, cwd=script_dir, check=True)
        print("\nTraining completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)