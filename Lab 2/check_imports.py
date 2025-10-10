try:
    import torch
    print("PyTorch imported successfully")
    print(f"PyTorch version: {torch.__version__}")
    
    import torchvision
    print("Torchvision imported successfully")
    print(f"Torchvision version: {torchvision.__version__}")
    
    from torchvision.io import decode_image
    print("decode_image imported successfully")
    
except Exception as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()