"""
Minimal test with existing images to verify CustomDataset works
"""
import os
import sys
sys.path.append('.')

# Create a test annotation file with an existing image
test_content = 'Abyssinian_1.jpg,"(100, 150)"'
with open('test_minimal.txt', 'w') as f:
    f.write(test_content)

print("Created test annotation file:")
print(test_content)

# Test the dataset
try:
    from train import CustomDataset
    
    # Create dataset with our test file
    dataset = CustomDataset('test_minimal.txt', 'images/')
    print(f"\nDataset created successfully!")
    print(f"Dataset size: {len(dataset)}")
    
    # Load the sample
    image, label = dataset[0]
    print(f"\nLoaded sample:")
    print(f"Image shape: {image.shape}")
    print(f"Label: {label}")
    
    # Verify requirements
    assert image.shape[0] == 3, f"Expected 3 channels, got {image.shape[0]}"
    assert image.shape[1] == 227, f"Expected height 227, got {image.shape[1]}"
    assert image.shape[2] == 227, f"Expected width 227, got {image.shape[2]}"
    
    print("\n✅ SUCCESS! CustomDataset is working correctly:")
    print(f"- Image transformed to {list(image.shape)}")
    print(f"- Coordinates transformed to {label.tolist()}")
    print("- Ready for training!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Clean up
    if os.path.exists('test_minimal.txt'):
        os.remove('test_minimal.txt')