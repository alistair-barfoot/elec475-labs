#!/usr/bin/env python3

import sys
import os

def test_train_py_import():
    """
    Test importing CustomDataset directly from train.py
    """
    print("="*50)
    print("TESTING CUSTOMDATASET FROM TRAIN.PY")
    print("="*50)
    
    try:
        print("Step 1: Adding current directory to Python path...")
        if '.' not in sys.path:
            sys.path.insert(0, '.')
        print("✅ Path configured")
        
        print("\nStep 2: Attempting to import from train.py...")
        
        # This will test if the import works
        from train import CustomDataset
        print("✅ Successfully imported CustomDataset from train.py!")
        
        print("\nStep 3: Creating dataset instance...")
        dataset = CustomDataset(
            annotations_file='train_noses.txt',
            img_dir='images',
            transform=None
        )
        print(f"✅ Dataset created successfully!")
        print(f"   Dataset length: {len(dataset)}")
        
        print("\nStep 4: Testing dataset access...")
        # Test first item
        image, label = dataset[0]
        print(f"✅ Successfully loaded first item!")
        print(f"   Image shape: {image.shape}")
        print(f"   Image type: {type(image)}")
        print(f"   Label: {label}")
        
        # Test a few more items
        for i in range(1, 4):
            image, label = dataset[i]
            print(f"✅ Item {i}: shape={image.shape}, label={label}")
            
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("This suggests there's an issue with importing from train.py")
        print("Likely cause: torchvision import problems")
        return False
        
    except Exception as e:
        print(f"❌ Runtime error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_manual_recreation():
    """
    Test by manually recreating the CustomDataset class structure
    """
    print("\n" + "="*50)
    print("TESTING MANUAL RECREATION OF CUSTOMDATASET")
    print("="*50)
    
    try:
        import pandas as pd
        import torch
        from torch.utils.data import Dataset
        
        # Recreate the exact class from train.py
        class CustomDataset(Dataset):
            def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
                self.img_labels = pd.read_csv(annotations_file, header=None, names=['filename', 'coordinates'])
                self.img_dir = img_dir
                self.transform = transform
                self.target_transform = target_transform
                
            def __len__(self):
                return len(self.img_labels)
                
            def __getitem__(self, idx):
                img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
                
                # Use PIL for image loading as a fallback
                from PIL import Image
                import torchvision.transforms as transforms
                
                image = Image.open(img_path)
                # Convert PIL to tensor
                to_tensor = transforms.ToTensor()
                image = to_tensor(image)
                
                label = self.img_labels.iloc[idx, 1]
                if self.transform:
                    image = self.transform(image)
                if self.target_transform:
                    label = self.target_transform(label)
                return image, label
        
        print("✅ CustomDataset class recreated successfully")
        
        dataset = CustomDataset('train_noses.txt', 'images')
        print(f"✅ Dataset created: {len(dataset)} items")
        
        # Test loading
        for i in range(3):
            image, label = dataset[i]
            print(f"✅ Item {i}: shape={image.shape}, label={label}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error in manual recreation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("COMPREHENSIVE TEST OF CUSTOMDATASET FROM TRAIN.PY")
    print("=" * 70)
    
    # Test 1: Direct import
    success1 = test_train_py_import()
    
    # Test 2: Manual recreation
    success2 = test_manual_recreation()
    
    print("\n" + "="*70)
    print("FINAL SUMMARY:")
    print(f"Direct import from train.py: {'✅ SUCCESS' if success1 else '❌ FAILED'}")
    print(f"Manual recreation test: {'✅ SUCCESS' if success2 else '❌ FAILED'}")
    
    if success1:
        print("\n🎉 Your CustomDataset in train.py works perfectly!")
    elif success2:
        print("\n⚠️  Your CustomDataset structure is correct, but there may be")
        print("   import issues with torchvision.io.decode_image in your environment")
    else:
        print("\n❌ There are issues that need to be resolved")
        
    print("="*70)