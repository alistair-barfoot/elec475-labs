import pandas as pd
import os
from torchvision.io import decode_image

# Test reading the CSV file properly and loading an image
try:
    # Read without headers and provide proper column names
    df = pd.read_csv('train_noses.txt', header=None, names=['filename', 'coordinates'])
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    print("First 5 rows:")
    print(df.head())
    
    # Test first image
    first_img = df.iloc[0, 0]
    img_path = os.path.join('images', first_img)
    print(f"First image path: {img_path}")
    print(f"File exists: {os.path.exists(img_path)}")
    
    # Try to load the image
    with open(img_path, 'rb') as f:
        image = decode_image(f.read())
    print(f"Image loaded successfully! Shape: {image.shape}")
    
    # Test label
    label = df.iloc[0, 1]
    print(f"Label: {label}, Type: {type(label)}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()