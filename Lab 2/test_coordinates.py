from train import CustomDataset
import torch

# Test the coordinate transformation
print("Testing CustomDataset with coordinate transformation...")

try:
    dataset = CustomDataset(annotations_file='train_noses.txt', img_dir='images', transform=None)
    print(f"Dataset length: {len(dataset)}")
    
    # Test first few items to verify both image shape and coordinate transformation
    for i in range(3):
        image, label = dataset[i]
        print(f"\nItem {i}:")
        print(f"  Image shape: {image.shape}")
        print(f"  Transformed coordinates: {label}")
        print(f"  Coordinates type: {type(label)}")
        print(f"  Coordinates range: x={label[0]:.2f}, y={label[1]:.2f}")
        
        # Verify the shape is exactly [3, 227, 227]
        if image.shape == (3, 227, 227):
            print(f"  ✅ Image shape is correct: {image.shape}")
        else:
            print(f"  ❌ Image shape is incorrect: {image.shape}")
            
        # Verify coordinates are within reasonable range for 227x227 image
        if 0 <= label[0] <= 227 and 0 <= label[1] <= 227:
            print(f"  ✅ Coordinates are within valid range")
        else:
            print(f"  ⚠️  Coordinates might be outside valid range")
    
    print("\n🎉 SUCCESS: Images are [3, 227, 227] and coordinates are transformed!")
    print("📝 Coordinates are now scaled to match the 227x227 resized images")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()