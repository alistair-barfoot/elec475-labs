import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import argparse
from model import snoutNet

class CustomDataset(Dataset):
    """Custom dataset class for loading images and labels"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # Assuming data is organized in folders like: data_dir/class_0/, data_dir/class_1/
        self.dataset = ImageFolder(root=data_dir, transform=transform)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

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

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, device, save_path='snoutNet_best.pth'):
    """Train the model and return training history"""
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    
    print(f"Training on device: {device}")
    print(f"Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (inputs, targets) in enumerate(train_pbar):
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
            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct_train / total_train:.2f}%'
            })
        
        # Calculate training metrics
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * correct_val / total_val:.2f}%'
                })
        
        # Calculate validation metrics
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        # Step the scheduler
        scheduler.step(epoch_val_loss)
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
        print(f'  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 50)
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_losses': train_losses,
                'train_accuracies': train_accuracies,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies
            }, save_path)
            print(f'New best model saved with validation accuracy: {best_val_acc:.2f}%')
    
    print(f'Training completed! Best validation accuracy: {best_val_acc:.2f}%')
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }

def plot_training_history(history, save_path='training_plots.png'):
    """Plot training and validation loss and accuracy"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(history['train_losses'], label='Training Loss', color='blue')
    ax1.plot(history['val_losses'], label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(history['train_accuracies'], label='Training Accuracy', color='blue')
    ax2.plot(history['val_accuracies'], label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Training plots saved to {save_path}')

def evaluate_model(model, test_loader, device):
    """Evaluate the model on test data"""
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(2))  # Assuming 2 classes
    class_total = list(0. for i in range(2))
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Per-class accuracy
            c = (predicted == targets).squeeze()
            for i in range(targets.size(0)):
                label = targets[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    print(f'Overall Test Accuracy: {100 * correct / total:.2f}%')
    
    # Print per-class accuracy
    for i in range(2):
        if class_total[i] > 0:
            print(f'Class {i} Accuracy: {100 * class_correct[i] / class_total[i]:.2f}%')

def main():
    parser = argparse.ArgumentParser(description='Train snoutNet model')
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='Path to the dataset directory')
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
    parser.add_argument('--train_split', type=float, default=0.8, 
                       help='Training split ratio (default: 0.8)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create transforms
    train_transform, val_transform = create_data_transforms()
    
    # Load dataset
    # Assuming your data is organized as: data_dir/train/, data_dir/val/, data_dir/test/
    # or you can modify this to split a single dataset
    try:
        train_dataset = ImageFolder(root=os.path.join(args.data_dir, 'train'), 
                                   transform=train_transform)
        val_dataset = ImageFolder(root=os.path.join(args.data_dir, 'val'), 
                                 transform=val_transform)
        test_dataset = ImageFolder(root=os.path.join(args.data_dir, 'test'), 
                                  transform=val_transform)
    except:
        # If the above structure doesn't exist, assume single directory and split
        print("Split directory structure not found. Using single directory and splitting...")
        full_dataset = ImageFolder(root=args.data_dir, transform=train_transform)
        
        # Split dataset
        train_size = int(args.train_split * len(full_dataset))
        val_size = (len(full_dataset) - train_size) // 2
        test_size = len(full_dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size])
        
        # Apply different transforms
        val_dataset.dataset.transform = val_transform
        test_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=2)
    
    print(f'Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}')
    
    # Initialize model
    model = snoutNet().to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
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
    print(f'Loaded best model from epoch {checkpoint["epoch"]} with validation accuracy: {checkpoint["best_val_acc"]:.2f}%')
    
    # Evaluate on test set
    print('\nEvaluating on test set...')
    evaluate_model(model, test_loader, device)

if __name__ == '__main__':
    main()