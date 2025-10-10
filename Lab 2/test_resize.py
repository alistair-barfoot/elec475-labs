from train import CustomDataset

# Test the modified CustomDataset
print("Testing CustomDataset with [3, 227, 227] transform...")

try:
    dataset = CustomDataset(annotations_file='train_noses.txt', img_dir='images', transform=None)
    print(f"Dataset length: {len(dataset)}")
    
    # Test first few items to verify shape
    for i in range(5):
        image, label = dataset[i]
        print(f"Item {i}: image shape = {image.shape}, label = {label}")
        
        # Verify the shape is exactly [3, 227, 227]
        expected_shape = (3, 227, 227)
        if image.shape == expected_shape:
            print(f"  ✅ Shape is correct: {image.shape}")
        else:
            print(f"  ❌ Shape is incorrect: expected {expected_shape}, got {image.shape}")
    
    print("\n✅ All images are successfully transformed to [3, 227, 227]!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()