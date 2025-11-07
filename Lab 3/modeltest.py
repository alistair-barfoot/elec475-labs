"""
Comprehensive Model Testing Suite
Tests model architecture, data pipeline, gradients, and training setup
without running full training loop.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

from MobileNet_model import MBV3SmallSeg
from dataloader import get_voc_dataloader


def test_model_architecture():
    """Test model creation and basic architecture."""
    print("="*60)
    print("🏗️  TESTING MODEL ARCHITECTURE")
    print("="*60)
    
    # Test model creation
    try:
        model = MBV3SmallSeg(
            num_classes=21,
            backbone_pretrained=True,
            input_size=(256, 256),
            dropout=0.1
        )
        print("✅ Model created successfully")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return None
    
    # Check parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"📊 Parameter Analysis:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {frozen_params:,}")
    
    if frozen_params > 0:
        print(f"⚠️  WARNING: {frozen_params:,} parameters are frozen!")
    
    # Test forward pass with dummy input
    try:
        model.eval()
        dummy_input = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ Forward pass successful")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        
        expected_shape = (2, 21, 256, 256)
        if output.shape == expected_shape:
            print(f"✅ Output shape correct: {output.shape}")
        else:
            print(f"❌ Output shape incorrect. Expected: {expected_shape}, Got: {output.shape}")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return None
    
    return model


def test_data_pipeline():
    """Test data loading and preprocessing."""
    print("\n" + "="*60)
    print("📊 TESTING DATA PIPELINE")
    print("="*60)
    
    try:
        # Test train dataloader
        train_loader = get_voc_dataloader(
            root='./archive',
            image_set='train',
            img_size=(256, 256),
            batch_size=4,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        # Test val dataloader
        val_loader = get_voc_dataloader(
            root='./archive',
            image_set='val',
            img_size=(256, 256),
            batch_size=4,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        print(f"✅ Dataloaders created successfully")
        print(f"   Train samples: {len(train_loader.dataset)}")
        print(f"   Val samples: {len(val_loader.dataset)}")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        
    except Exception as e:
        print(f"❌ Dataloader creation failed: {e}")
        return None, None
    
    # Test data loading
    try:
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        train_images, train_masks = train_batch
        val_images, val_masks = val_batch
        
        print(f"✅ Data loading successful")
        print(f"   Train batch - Images: {train_images.shape}, Masks: {train_masks.shape}")
        print(f"   Val batch - Images: {val_images.shape}, Masks: {val_masks.shape}")
        
        # Check data types and ranges
        print(f"\n🔍 Data Analysis:")
        print(f"   Image dtype: {train_images.dtype}, range: [{train_images.min():.3f}, {train_images.max():.3f}]")
        print(f"   Mask dtype: {train_masks.dtype}, range: [{train_masks.min()}, {train_masks.max()}]")
        
        # Check mask values
        unique_mask_values = torch.unique(train_masks)
        print(f"   Unique mask values: {unique_mask_values.numpy()}")
        
        # Verify mask values are correct for VOC (0-20 classes + 255 ignore)
        valid_values = set(range(21)) | {255}
        mask_values = set(unique_mask_values.numpy())
        
        if mask_values.issubset(valid_values):
            print(f"✅ Mask values are valid VOC format")
        else:
            invalid = mask_values - valid_values
            print(f"❌ Invalid mask values found: {invalid}")
        
        # Check for class distribution
        mask_counts = torch.bincount(train_masks.flatten(), minlength=256)
        non_zero_classes = torch.nonzero(mask_counts).flatten()
        print(f"   Classes present in batch: {len(non_zero_classes)} classes")
        
        return train_loader, val_loader
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None, None


def test_model_predictions(model, dataloader):
    """Test model predictions and initialization."""
    print("\n" + "="*60)
    print("🔮 TESTING MODEL PREDICTIONS")
    print("="*60)
    
    if model is None or dataloader is None:
        print("❌ Cannot test predictions - model or dataloader failed")
        return
    
    model.eval()
    images, masks = next(iter(dataloader))
    
    with torch.no_grad():
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
    
    # Analyze predictions
    unique_preds = torch.unique(preds)
    print(f"📊 Prediction Analysis:")
    print(f"   Model predicting {len(unique_preds)} unique classes: {unique_preds.numpy()}")
    
    # Check prediction distribution
    pred_counts = torch.bincount(preds.flatten(), minlength=21)
    most_frequent_class = torch.argmax(pred_counts)
    most_frequent_percentage = (pred_counts[most_frequent_class] / preds.numel() * 100).item()
    
    print(f"   Most frequent prediction: class {most_frequent_class} ({most_frequent_percentage:.1f}% of pixels)")
    
    # Warning checks
    if len(unique_preds) == 1:
        print(f"⚠️  WARNING: Model predicting only ONE class - bad initialization!")
    elif len(unique_preds) < 5:
        print(f"⚠️  WARNING: Model predicting very few classes - may need better initialization")
    elif most_frequent_percentage > 95:
        print(f"⚠️  WARNING: Model heavily biased towards one class")
    else:
        print(f"✅ Prediction diversity looks reasonable")
    
    # Check logit statistics
    logit_mean = logits.mean().item()
    logit_std = logits.std().item()
    logit_min = logits.min().item()
    logit_max = logits.max().item()
    
    print(f"\n🔢 Logit Statistics:")
    print(f"   Mean: {logit_mean:.4f}, Std: {logit_std:.4f}")
    print(f"   Range: [{logit_min:.4f}, {logit_max:.4f}]")
    
    if abs(logit_mean) > 5:
        print(f"⚠️  WARNING: Logit mean is large - may indicate initialization issues")
    if logit_std < 0.1:
        print(f"⚠️  WARNING: Low logit variance - model may not be expressive enough")


def test_gradient_flow(model, dataloader):
    """Test gradient computation and flow."""
    print("\n" + "="*60)
    print("🌊 TESTING GRADIENT FLOW")
    print("="*60)
    
    if model is None or dataloader is None:
        print("❌ Cannot test gradients - model or dataloader failed")
        return
    
    # Setup for gradient test
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    images, masks = next(iter(dataloader))
    
    # Forward pass
    model.zero_grad()
    logits = model(images)
    loss = criterion(logits, masks)
    
    print(f"📊 Loss Analysis:")
    print(f"   Loss value: {loss.item():.6f}")
    
    # Expected loss for random predictions: -log(1/21) ≈ 3.04
    expected_random_loss = -np.log(1/21)
    print(f"   Expected random loss: {expected_random_loss:.6f}")
    
    if loss.item() > expected_random_loss * 2:
        print(f"⚠️  WARNING: Loss is much higher than random - check data/model")
    elif loss.item() < 0.1:
        print(f"⚠️  WARNING: Loss is very low - may indicate issues")
    
    # Backward pass
    loss.backward()
    
    # Analyze gradients
    gradient_stats = []
    zero_grad_count = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += 1
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_stats.append(grad_norm)
                if grad_norm == 0:
                    zero_grad_count += 1
            else:
                print(f"⚠️  WARNING: No gradient for parameter: {name}")
    
    if gradient_stats:
        grad_mean = np.mean(gradient_stats)
        grad_std = np.std(gradient_stats)
        grad_min = np.min(gradient_stats)
        grad_max = np.max(gradient_stats)
        
        print(f"\n🌊 Gradient Statistics:")
        print(f"   Parameters with gradients: {len(gradient_stats)}/{total_params}")
        print(f"   Gradient norm - Mean: {grad_mean:.6f}, Std: {grad_std:.6f}")
        print(f"   Gradient range: [{grad_min:.6f}, {grad_max:.6f}]")
        print(f"   Parameters with zero gradients: {zero_grad_count}")
        
        # Warning checks
        if grad_mean < 1e-6:
            print(f"⚠️  WARNING: Very small gradients - vanishing gradient problem!")
        elif grad_mean > 1e2:
            print(f"⚠️  WARNING: Very large gradients - exploding gradient problem!")
        elif zero_grad_count > total_params * 0.1:
            print(f"⚠️  WARNING: Many parameters have zero gradients")
        else:
            print(f"✅ Gradient magnitudes look reasonable")
    else:
        print(f"❌ No gradients computed!")


def test_optimizer_setup():
    """Test optimizer configuration."""
    print("\n" + "="*60)
    print("⚙️  TESTING OPTIMIZER SETUP")
    print("="*60)
    
    # Create model
    model = MBV3SmallSeg(num_classes=21, backbone_pretrained=True, input_size=(256, 256))
    
    # Test current configuration from training_loop.py
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.0  # From your current settings
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
    
    print(f"📊 Optimizer Configuration:")
    print(f"   Type: {type(optimizer).__name__}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Weight Decay: {WEIGHT_DECAY}")
    print(f"   Betas: {optimizer.param_groups[0]['betas']}")
    
    # Check parameter groups
    param_groups = len(optimizer.param_groups)
    total_params_in_optimizer = sum(len(group['params']) for group in optimizer.param_groups)
    total_model_params = len(list(model.parameters()))
    
    print(f"   Parameter groups: {param_groups}")
    print(f"   Parameters in optimizer: {total_params_in_optimizer}")
    print(f"   Total model parameters: {total_model_params}")
    
    if total_params_in_optimizer != total_model_params:
        print(f"⚠️  WARNING: Mismatch in parameter counts!")
    else:
        print(f"✅ All model parameters are in optimizer")


def run_comprehensive_test():
    """Run all tests."""
    print("🔬 COMPREHENSIVE MODEL TESTING SUITE")
    print("="*60)
    
    # Test 1: Model Architecture
    model = test_model_architecture()
    
    # Test 2: Data Pipeline
    train_loader, val_loader = test_data_pipeline()
    
    # Test 3: Model Predictions
    test_model_predictions(model, val_loader)
    
    # Test 4: Gradient Flow
    test_gradient_flow(model, train_loader)
    
    # Test 5: Optimizer Setup
    test_optimizer_setup()
    
    print("\n" + "="*60)
    print("🏁 TESTING COMPLETE")
    print("="*60)
    print("\n💡 Next steps based on results:")
    print("   - If any ❌ or ⚠️  appear above, address those issues first")
    print("   - If all tests pass ✅, the issue may be in training hyperparameters")
    print("   - Pay special attention to gradient flow and prediction diversity")


if __name__ == '__main__':
    run_comprehensive_test()