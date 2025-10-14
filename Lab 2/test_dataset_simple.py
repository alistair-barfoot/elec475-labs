import os
import pandas as pd
import numpy as np
import cv2
from torchvision.transforms import v2
from dataset import CustomDataset

def test_dataset(start_i=0):
    # Initialize the dataset
    transform = v2.Compose([
        v2.Resize((227, 227)),
        v2.RandomHorizontalFlip(p=0.5), 
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=(-90,90)),
        v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        v2.ToTensor()],
    )
    
    dataset = CustomDataset(annotations_file='train_noses.txt', img_dir='images', transform=transform)
            
    for i in range(start_i, start_i+3):
        image, label = dataset[i]
        filename = dataset.img_labels.iloc[i, 0]
        print(f"Item {i}: file={filename}, shape={image.shape}, label={label}")
        
        # Convert tensor image back to numpy array for OpenCV
        # Image is in format [3, 227, 227], need to convert to [227, 227, 3]
        img_numpy = image.numpy().transpose(1, 2, 0)  # [H, W, C]
        img_numpy = (img_numpy * 255).astype(np.uint8)  # Convert to 0-255 range
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
        
        # Draw circle at nose coordinates
        nose_x = int(label[0].item())
        nose_y = int(label[1].item())
        print("="*75)
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

if __name__ == "__main__":
    test_dataset()