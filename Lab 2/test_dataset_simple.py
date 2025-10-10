import os
import pandas as pd
import numpy as np
import cv2
from train import CustomDataset

# Test dataset loading using the CustomDataset class structure from train.py
def test_dataset_simple():
    print("=" * 60)
    print("TESTING CUSTOMDATASET FROM TRAIN.PY")
    print("=" * 60)
    
    try:
        # Try to import directly from train.py first
        print("Attempt 1: Direct import from train.py...")
        try:
            print("✅ Direct import successful!")
            
            dataset = CustomDataset(annotations_file='train_noses.txt', img_dir='images', transform=None)
            print(f"✅ Dataset created successfully. Length: {len(dataset)}")
            
            # Test loading first few items
            for i in range(3):
                image, label = dataset[i]
                filename = dataset.img_labels.iloc[i, 0]
                print(f"✅ Item {i}: file={filename}, shape={image.shape}, label={label}")
                cv2.imshow("Test Image", cv2.cvtColor(image.numpy().transpose(1, 2, 0), cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)  # Display each image until a key is pressed
            
            return True, "Direct import from train.py works perfectly!"
            
        except Exception as e:
            print(f"❌ Direct import failed: {e}")
            print("Likely cause: torchvision.io.decode_image import issues")
            
        
    except Exception as e:
        print(f"❌ Error in testing: {e}")
        import traceback
        traceback.print_exc()
        return False, f"Error: {e}"

if __name__ == "__main__":
    success, message = test_dataset_simple()
    
    print("\n" + "="*60)
    
    if success:
        print("Done\n")
        
    else:
        print("Fail")
        print(f"Error details: {message}")
        
    print("="*60)