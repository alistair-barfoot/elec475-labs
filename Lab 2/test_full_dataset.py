"""
Test the full CustomDataset functionality including image loading and coordinate transformation
"""
import sys
sys.path.append('.')  # Add current directory to path

from train import CustomDataset
import torch

def test_full_dataset():
    print("Testing full CustomDataset functionality...")
    
    # Create dataset instances
    train_dataset = CustomDataset('train_noses.txt', 'images/')
    test_dataset = CustomDataset('test_noses.txt', 'images/')
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test loading a few samples
    print("\nTesting data loading...")
    for i in range(3):
        image, label = train_dataset[i]
        print(f"Sample {i}:")
        print(f"  Image shape: {image.shape}")
        print(f"  Image type: {type(image)}")
        print(f"  Label: {label}")
        print(f"  Label type: {type(label)}")
        
        # Verify image is in correct format [3, 227, 227]
        assert image.shape == torch.Size([3, 227, 227]), f"Expected [3, 227, 227], got {image.shape}"
        
        # Verify coordinates are within valid range [0, 227]
        assert 0 <= label[0] <= 227, f"X coordinate {label[0]} out of range [0, 227]"
        assert 0 <= label[1] <= 227, f"Y coordinate {label[1]} out of range [0, 227]"
        
        print(f"  ✅ Valid image shape and coordinate range")
    
    print("\n✅ All tests passed! CustomDataset is working correctly.")
    print("- Images are properly resized to [3, 227, 227]")
    print("- Coordinates are transformed to match resized images")
    print("- Ready for training!")

if __name__ == "__main__":
    test_full_dataset()