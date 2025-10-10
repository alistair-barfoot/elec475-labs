"""
Test CustomDataset with nose coordinate visualization
"""
import os
import numpy as np
import cv2
from train import CustomDataset

def test_with_visualization():
    print("Testing CustomDataset with nose coordinate visualization...")
    
    try:
        # Create dataset
        dataset = CustomDataset('train_noses.txt', 'images/')
        print(f"Dataset length: {len(dataset)}")
        
        # Test first few samples
        for i in range(min(3, len(dataset))):
            print(f"\n--- Sample {i+1} ---")
            
            # Load image and label
            image, label = dataset[i]
            filename = dataset.img_labels.iloc[i, 0]
            
            print(f"File: {filename}")
            print(f"Image shape: {image.shape}")
            print(f"Nose coordinates: ({label[0]:.1f}, {label[1]:.1f})")
            
            # Convert tensor to displayable image
            img_numpy = image.numpy().transpose(1, 2, 0)  # [H, W, C]
            img_numpy = (img_numpy * 255).astype(np.uint8)  # Convert to 0-255
            img_bgr = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
            
            # Draw nose marker
            nose_x = int(label[0].item())
            nose_y = int(label[1].item())
            
            # Draw filled green circle
            cv2.circle(img_bgr, (nose_x, nose_y), 5, (0, 255, 0), -1)
            # Draw red border for visibility
            cv2.circle(img_bgr, (nose_x, nose_y), 8, (0, 0, 255), 2)
            
            # Save image instead of displaying (to avoid hanging)
            output_path = f"nose_test_{i+1}_{filename}"
            cv2.imwrite(output_path, img_bgr)
            print(f"✅ Saved visualization: {output_path}")
            
        print(f"\n✅ SUCCESS! Generated {min(3, len(dataset))} test images with nose markers.")
        print("Check the generated images to see the nose detection visualization!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_with_visualization()