import pandas as pd
import os

# Test reading the CSV file
try:
    df = pd.read_csv('train_noses.txt')
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    print("First 5 rows:")
    print(df.head())
    print(f"Columns: {df.columns.tolist()}")
    
    # Test first image path
    first_img = df.iloc[0, 0]
    img_path = os.path.join('images', first_img)
    print(f"First image path: {img_path}")
    print(f"File exists: {os.path.exists(img_path)}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()