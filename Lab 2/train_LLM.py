import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import ast
from model import snoutNet

class NoseDataset(Dataset):
    """Custom dataset class for loading images and nose coordinate labels"""
    def __init__(self, images_dir, labels_file, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.data = []
        
        # Parse the labels file
        with open(labels_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Parse format: filename,"(x, y)"
                    parts = line.split(',', 1)
                    if len(parts) == 2:
                        filename = parts[0].strip()
                        coord_str = parts[1].strip().strip('"')
                        # Parse coordinates from string "(x, y)"
                        try:
                            coords = ast.literal_eval(coord_str)
                            if isinstance(coords, tuple) and len(coords) == 2:
                                x, y = coords
                                self.data.append((filename, float(x), float(y)))
                        except (ValueError, SyntaxError) as e:
                            print(f"Warning: Could not parse coordinates for {filename}: {coord_str}")
                            continue
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        filename, x, y = self.data[idx]
        
        # Load image
        image_path = os.path.join(self.images_dir, filename)
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (227, 227), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Convert coordinates to tensor
        coordinates = torch.tensor([x, y], dtype=torch.float32)
        
        return image, coordinates

def create_data_transforms():
    """Create data transforms for training and validation"""
    train_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def validate_dataset(dataset, sample_size=10):
    """Validate dataset by checking a sample of images and labels"""
    print(f"Dataset validation - Total samples: {len(dataset)}")
    
    if len(dataset) == 0:
        print("ERROR: Dataset is empty!")
        return False
        
    # Check a sample of data
    sample_indices = np.random.choice(len(dataset), min(sample_size, len(dataset)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        try:
            image, coords = dataset[idx]
            filename = dataset.data[idx][0]
            
            print(f"Sample {i+1}: {filename}")
            print(f"  Image shape: {image.shape}")
            print(f"  Coordinates: ({coords[0].item():.1f}, {coords[1].item():.1f})")
            print(f"  Image range: [{image.min().item():.3f}, {image.max().item():.3f}]")
            
            # Check if coordinates seem reasonable (assuming image size 227x227)
            if coords[0] < 0 or coords[1] < 0:
                print(f"  WARNING: Negative coordinates detected!")
            if coords[0] > 1000 or coords[1] > 1000:  # Reasonable upper bound
                print(f"  WARNING: Very large coordinates detected!")
                
        except Exception as e:
            print(f"Sample {i+1}: ERROR - {e}")
            return False
    
    return True

def visualize_samples(dataset, num_samples=5, save_path='dataset_samples.png'):
    """Visualize sample images with nose coordinates"""
    if len(dataset) < num_samples:
        num_samples = len(dataset)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    if num_samples == 1:
        axes = [axes]
    
    sample_indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(sample_indices):
        image, coords = dataset[idx]
        filename = dataset.data[idx][0]
        
        # Convert tensor to numpy and denormalize
        image_np = image.permute(1, 2, 0).numpy()
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = image_np * std + mean
        image_np = np.clip(image_np, 0, 1)
        
        axes[i].imshow(image_np)
        
        # Scale coordinates to image size (227x227)
        x_scaled = coords[0].item() * (227 / 227)  # Adjust if original image size differs
        y_scaled = coords[1].item() * (227 / 227)
        
        # Plot nose location
        axes[i].plot(x_scaled, y_scaled, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=2)
        axes[i].set_title(f'{filename}\n({coords[0].item():.0f}, {coords[1].item():.0f})', fontsize=8)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Sample visualizations saved to {save_path}')

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, device, save_path='snoutNet_best.pth'):
    """Train the model and return training history"""
    
    train_losses = []
    train_errors = []  # Mean coordinate error instead of accuracy
    val_losses = []
    val_errors = []
    best_val_error = float('inf')  # Lower is better for error
    
    print(f"Training on device: {device}")
    print(f"Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        total_error = 0.0
        num_samples = 0
        
        train_pbar = enumerate(train_loader)
        for batch_idx, (inputs, targets) in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            
            # Calculate coordinate error (Euclidean distance)
            with torch.no_grad():
                error = torch.sqrt(torch.sum((outputs - targets) ** 2, dim=1)).mean()
                total_error += error.item() * inputs.size(0)
                num_samples += inputs.size(0)
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                current_error = total_error / num_samples
                print(f'  Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.4f}, Error={current_error:.2f}px')
        
        # Calculate training metrics
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_error = total_error / num_samples
        train_losses.append(epoch_train_loss)
        train_errors.append(epoch_train_error)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        total_val_error = 0.0
        num_val_samples = 0
        
        with torch.no_grad():
            val_batch_count = 0
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                
                # Calculate coordinate error
                error = torch.sqrt(torch.sum((outputs - targets) ** 2, dim=1)).mean()
                total_val_error += error.item() * inputs.size(0)
                num_val_samples += inputs.size(0)
                
                # Print progress every 10 batches
                if val_batch_count % 10 == 0:
                    current_val_error = total_val_error / num_val_samples
                    print(f'  Val Batch {val_batch_count}/{len(val_loader)}: Loss={loss.item():.4f}, Error={current_val_error:.2f}px')
                val_batch_count += 1
        
        # Calculate validation metrics
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_error = total_val_error / num_val_samples
        val_losses.append(epoch_val_loss)
        val_errors.append(epoch_val_error)
        
        # Step the scheduler
        scheduler.step(epoch_val_loss)
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {epoch_train_loss:.4f}, Train Error: {epoch_train_error:.2f}px')
        print(f'  Val Loss: {epoch_val_loss:.4f}, Val Error: {epoch_val_error:.2f}px')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 50)
        
        # Save best model (lower error is better)
        if epoch_val_error < best_val_error:
            best_val_error = epoch_val_error
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_error': best_val_error,
                'train_losses': train_losses,
                'train_errors': train_errors,
                'val_losses': val_losses,
                'val_errors': val_errors
            }, save_path)
            print(f'New best model saved with validation error: {best_val_error:.2f}px')
    
    print(f'Training completed! Best validation error: {best_val_error:.2f}px')
    
    return {
        'train_losses': train_losses,
        'train_errors': train_errors,
        'val_losses': val_losses,
        'val_errors': val_errors
    }

def plot_training_history(history, save_path='training_plots.png'):
    """Plot training and validation loss and error"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(history['train_losses'], label='Training Loss', color='blue')
    ax1.plot(history['val_losses'], label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot errors
    ax2.plot(history['train_errors'], label='Training Error', color='blue')
    ax2.plot(history['val_errors'], label='Validation Error', color='red')
    ax2.set_title('Training and Validation Error')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Error (pixels)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Training plots saved to {save_path}')

def evaluate_model(model, test_loader, device):
    """Evaluate the model on test data"""
    model.eval()
    total_error = 0.0
    num_samples = 0
    all_errors = []
    
    with torch.no_grad():
        batch_count = 0
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Calculate coordinate errors (Euclidean distance)
            errors = torch.sqrt(torch.sum((outputs - targets) ** 2, dim=1))
            all_errors.extend(errors.cpu().numpy())
            total_error += errors.sum().item()
            num_samples += inputs.size(0)
            
            # Print progress every 10 batches
            if batch_count % 10 == 0:
                print(f'  Evaluating batch {batch_count}/{len(test_loader)}')
            batch_count += 1
    
    mean_error = total_error / num_samples
    all_errors = np.array(all_errors)
    
    print(f'Test Results:')
    print(f'  Mean Error: {mean_error:.2f} pixels')
    print(f'  Median Error: {np.median(all_errors):.2f} pixels')
    print(f'  Std Error: {np.std(all_errors):.2f} pixels')
    print(f'  Min Error: {np.min(all_errors):.2f} pixels')
    print(f'  Max Error: {np.max(all_errors):.2f} pixels')
    print(f'  95th percentile: {np.percentile(all_errors, 95):.2f} pixels')
    
    return mean_error, all_errors

def main():
    parser = argparse.ArgumentParser(description='Train snoutNet model for nose detection')
    parser.add_argument('--images_dir', type=str, required=True, 
                       help='Path to the images directory')
    parser.add_argument('--train_labels', type=str, required=True,
                       help='Path to training labels file (e.g., train_noses.txt)')
    parser.add_argument('--test_labels', type=str, required=True,
                       help='Path to test labels file (e.g., test_noses.txt)')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--save_path', type=str, default='snoutNet_best.pth', 
                       help='Path to save the best model')
    parser.add_argument('--val_split', type=float, default=0.2, 
                       help='Validation split ratio from training data (default: 0.2)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create transforms
    train_transform, val_transform = create_data_transforms()
    
    # Load datasets
    print("Loading datasets...")
    full_train_dataset = NoseDataset(args.images_dir, args.train_labels, transform=train_transform)
    test_dataset = NoseDataset(args.images_dir, args.test_labels, transform=val_transform)
    
    # Validate datasets
    print("\nValidating training dataset...")
    if not validate_dataset(full_train_dataset):
        print("Training dataset validation failed!")
        return
    
    print("\nValidating test dataset...")
    if not validate_dataset(test_dataset):
        print("Test dataset validation failed!")
        return
    
    # Split training data into train and validation
    train_size = int((1 - args.val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size])
    
    # Apply validation transform to validation set
    # Note: Since we're using random_split, we need to handle transforms differently
    # The validation set will still use the training transform from the original dataset
    # For a more sophisticated approach, you could create separate datasets
    
    print(f'Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=2)
    
    # Visualize some samples
    print("\nVisualizing sample data...")
    visualize_samples(full_train_dataset, num_samples=5)
    
    # Initialize model
    model = snoutNet().to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Define loss function and optimizer (MSE for regression)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, 
                          weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                    factor=0.5, patience=5, verbose=True)
    
    # Train the model
    history = train_model(model, train_loader, val_loader, criterion, optimizer, 
                         scheduler, args.epochs, device, args.save_path)
    
    # Plot training history
    plot_training_history(history)
    
    # Load best model and evaluate on test set
    checkpoint = torch.load(args.save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded best model from epoch {checkpoint["epoch"]} with validation error: {checkpoint["best_val_error"]:.2f}px')
    
    # Evaluate on test set
    print('\nEvaluating on test set...')
    test_error, all_errors = evaluate_model(model, test_loader, device)
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(all_errors, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Prediction Error (pixels)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors on Test Set')
    plt.axvline(test_error, color='red', linestyle='--', label=f'Mean Error: {test_error:.2f}px')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print('Error distribution plot saved to error_distribution.png')

if __name__ == '__main__':
    main()