import os
import pandas as pd
import numpy as np
import cv2
from dataset import CustomDataset

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
                
                # Convert tensor image back to numpy array for OpenCV
                # Image is in format [3, 227, 227], need to convert to [227, 227, 3]
                img_numpy = image.numpy().transpose(1, 2, 0)  # [H, W, C]
                img_numpy = (img_numpy * 255).astype(np.uint8)  # Convert to 0-255 range
                
                # Convert RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
                
                # Draw circle at nose coordinates
                nose_x = int(label[0].item())
                nose_y = int(label[1].item())
                print(f"   Drawing circle at nose position: ({nose_x}, {nose_y})")
                
                # Draw a green circle at the nose position
                cv2.circle(img_bgr, (nose_x, nose_y), 5, (0, 255, 0), -1)  # Filled green circle
                
                # Also draw a red border for better visibility
                cv2.circle(img_bgr, (nose_x, nose_y), 7, (0, 0, 255), 2)   # Red border circle
                
                # Display the image
                cv2.imshow(f"Nose Detection - {filename}", img_bgr)
                print(f"   Press any key to continue to next image...")
                cv2.waitKey(0)  # Wait for key press
                cv2.destroyAllWindows()  # Close the window
            
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