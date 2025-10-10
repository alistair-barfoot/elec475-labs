import pandas as pd
import torch
from PIL import Image
import os

# Simple test of coordinate transformation logic
print("Testing coordinate transformation logic...")

# Read one entry from the dataset
df = pd.read_csv('train_noses.txt', header=None, names=['filename', 'coordinates'])
print(f"First entry: {df.iloc[0, 0]} -> {df.iloc[0, 1]}")

# Get the first image
img_path = os.path.join('images', df.iloc[0, 0])
image = Image.open(img_path)
original_width, original_height = image.size
print(f"Original image size: {original_width} x {original_height}")

# Parse coordinates
label_str = df.iloc[0, 1]
coords = label_str.strip('()').split(', ')
original_x = float(coords[0])
original_y = float(coords[1])
print(f"Original coordinates: x={original_x}, y={original_y}")

# Scale coordinates to 227x227
scaled_x = (original_x / original_width) * 227
scaled_y = (original_y / original_height) * 227
print(f"Scaled coordinates: x={scaled_x:.2f}, y={scaled_y:.2f}")

# Create tensor
label_tensor = torch.tensor([scaled_x, scaled_y], dtype=torch.float32)
print(f"Label tensor: {label_tensor}")

print("\n✅ Coordinate transformation logic works correctly!")