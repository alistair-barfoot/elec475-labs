# ELEC 475 Lab 2 - snoutNet CNN for Nose Detection

This repository contains the implementation of **snoutNet**, a Convolutional Neural Network designed for animal nose detection and localization. The network processes 227x227x3 RGB images and predicts (x, y) coordinates of nose locations.

## 📁 Project Structure

```
Lab 2/
├── README.md               # This file
├── model.py               # snoutNet CNN architecture
├── train.py               # Updated training script for nose detection
├── validate_data.py       # Dataset validation and visualization
├── run_training.py        # Convenience script for training
├── train_noses.txt        # Training labels (image,coordinates)
├── test_noses.txt         # Test labels (image,coordinates)
├── images/                # Directory containing all image files
│   ├── beagle_145.jpg
│   ├── shiba_inu_136.jpg
│   └── ...
└── ELEC475_Lab2.pdf       # Lab assignment document
```

## 🎯 Task: Nose Detection

This implementation has been updated to perform **nose coordinate regression** instead of classification. The model:
- Takes RGB images of animals as input
- Outputs 2D coordinates (x, y) predicting nose location
- Uses MSE loss for regression training
- Reports accuracy in terms of pixel-level error

## 🧠 Model Architecture

**snoutNet** is a custom CNN with the following architecture:

### Network Layers:
1. **Conv1**: 3→64 channels, 3x3 kernel, stride=4, padding=1
   - Input: 227×227×3 → Output: 57×57×64
   - ReLU activation + MaxPool (3x3, stride=4, padding=1)

2. **Conv2**: 64→128 channels, 3x3 kernel, stride=4, padding=1
   - Input: 57×57×64 → Output: 15×15×128
   - ReLU activation + MaxPool (3x3, stride=4, padding=1)

3. **Conv3**: 128→256 channels, 3x3 kernel, stride=4, padding=1
   - Input: 15×15×128 → Output: 4×4×256
   - ReLU activation + MaxPool (3x3, stride=4, padding=1)

4. **FC1**: 4096→1024 (Fully Connected)
   - ReLU activation

5. **FC2**: 1024→1024 (Fully Connected)
   - ReLU activation

6. **FC3**: 1024→2 (Output layer for binary classification)

### Model Specifications:
- **Input Size**: 227×227×3 RGB images
- **Output**: 2 classes (binary classification)
- **Total Parameters**: ~16.8M parameters
- **Architecture Type**: Custom CNN with aggressive downsampling

## 🚀 Getting Started

### Prerequisites

Install the required dependencies:

```bash
pip install torch torchvision matplotlib numpy tqdm
```

### Dataset Organization

Your dataset should be organized in one of two ways:

**Option 1: Pre-split Dataset (Recommended)**
```
your_dataset/
├── train/
│   ├── class_0/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── class_1/
│       ├── image3.jpg
│       ├── image4.jpg
│       └── ...
├── val/
│   ├── class_0/
│   └── class_1/
└── test/
    ├── class_0/
    └── class_1/
```

**Option 2: Single Directory (Auto-split)**
```
your_dataset/
├── class_0/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── class_1/
    ├── image3.jpg
    ├── image4.jpg
    └── ...
```

## 🏋️ Training

### Basic Training Command

```bash
python train.py --data_dir "path/to/your/dataset"
```

### Advanced Training Options

```bash
python train.py \
    --data_dir "path/to/your/dataset" \
    --batch_size 64 \
    --epochs 100 \
    --learning_rate 0.0001 \
    --weight_decay 1e-4 \
    --save_path "my_model.pth" \
    --train_split 0.8
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_dir` | str | **Required** | Path to dataset directory |
| `--batch_size` | int | 32 | Batch size for training |
| `--epochs` | int | 50 | Number of training epochs |
| `--learning_rate` | float | 0.001 | Learning rate for optimizer |
| `--weight_decay` | float | 1e-4 | Weight decay for regularization |
| `--save_path` | str | snoutNet_best.pth | Path to save best model |
| `--train_split` | float | 0.8 | Training split ratio (for auto-split) |

## 📊 Training Features

The training script includes several advanced features:

### Data Augmentation
- **Training**: Random horizontal flips, rotations (±10°), color jittering
- **Validation/Test**: Only resizing and normalization
- **Normalization**: ImageNet standards (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Training Monitoring
- **Progress Bars**: Real-time training progress with loss and accuracy
- **Learning Rate Scheduling**: Reduces LR on validation loss plateau
- **Early Stopping**: Saves best model based on validation accuracy
- **Comprehensive Logging**: Tracks all metrics throughout training

### Automatic Visualization
- **Training Plots**: Loss and accuracy curves saved as high-resolution images
- **Real-time Metrics**: Live updates during training

### Model Checkpointing
The training script automatically saves the best model with:
- Model state dictionary
- Optimizer state
- Training history
- Best validation accuracy
- Epoch information

## 📈 Model Evaluation

After training, the script automatically:
1. Loads the best model checkpoint
2. Evaluates on the test set
3. Reports overall and per-class accuracy
4. Generates training history plots

### Sample Output
```
Training completed! Best validation accuracy: 95.32%
Loaded best model from epoch 43 with validation accuracy: 95.32%

Evaluating on test set...
Overall Test Accuracy: 94.76%
Class 0 Accuracy: 96.12%
Class 1 Accuracy: 93.41%
```

## 🔧 Using the Trained Model

### Loading a Trained Model

```python
import torch
from model import snoutNet

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = snoutNet().to(device)

# Load trained weights
checkpoint = torch.load('snoutNet_best.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model trained for {checkpoint['epoch']} epochs")
print(f"Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
```

### Making Predictions

```python
import torchvision.transforms as transforms
from PIL import Image

# Define the same transforms used during training
transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess an image
image = Image.open('path/to/your/image.jpg')
input_tensor = transform(image).unsqueeze(0).to(device)

# Make prediction
with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    predicted_class = torch.argmax(outputs, dim=1).item()

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {probabilities[0][predicted_class].item():.4f}")
```

## 🛠️ Customization

### Modifying the Model
To modify the snoutNet architecture, edit `model.py`. Key considerations:
- Maintain input size of 227×227×3
- Adjust the final fully connected layer if changing the number of classes
- Update the flattening dimension in `fc1` if changing convolutional layers

### Training Hyperparameters
Common hyperparameters to tune:
- **Learning Rate**: Start with 0.001, reduce if loss oscillates
- **Batch Size**: Increase if you have more GPU memory
- **Weight Decay**: Adjust for regularization (typical range: 1e-5 to 1e-3)
- **Epochs**: Monitor validation curves to avoid overfitting

## 📋 Requirements

- Python 3.7+
- PyTorch 1.9+
- torchvision
- matplotlib
- numpy
- tqdm
- PIL (Pillow)
## � Dataset Format for Nose Detection

### Label Files Format
The nose detection task uses coordinate labels in text files:

**train_noses.txt** and **test_noses.txt** format:
```
filename,"(x, y)"
```

Example:
```
beagle_145.jpg,"(198, 304)"
shiba_inu_136.jpg,"(182, 203)"
english_cocker_spaniel_181.jpg,"(145, 293)"
chihuahua_165.jpg,"(122, 122)"
```

### Usage Instructions

#### 1. Validate Dataset
Before training, check your data:
```bash
python validate_data.py --images_dir images --train_labels train_noses.txt --test_labels test_noses.txt
```

#### 2. Train the Model
Quick start with optimal settings:
```bash
python run_training.py
```

Or manually specify parameters:
```bash
python train.py \
  --images_dir images \
  --train_labels train_noses.txt \
  --test_labels test_noses.txt \
  --batch_size 16 \
  --epochs 25 \
  --learning_rate 0.0001
```

#### 3. Key Parameters
- `--images_dir`: Directory containing all image files
- `--train_labels`: Path to training coordinate labels
- `--test_labels`: Path to test coordinate labels  
- `--val_split`: Fraction of training data for validation (default: 0.2)

### Performance Metrics
- **Loss**: Mean Squared Error between predicted and actual coordinates
- **Error**: Mean Euclidean distance in pixels
- **Good Performance**: Mean error < 20-30 pixels

## 🐛 Troubleshooting

### Common Issues

1. **Dataset Validation Fails**
   ```bash
   python validate_data.py  # Check for missing images or parsing errors
   ```

2. **Large Coordinate Errors**
   - Reduce learning rate to 0.0001 or lower
   - Increase training epochs
   - Check coordinate label accuracy

3. **CUDA Out of Memory**
   ```bash
   python train.py --batch_size 8  # Reduce batch size
   ```

4. **Coordinate Parse Errors**
   - Ensure coordinate format is exactly: `filename,"(x, y)"`
   - Check for special characters in filenames
   - Verify all referenced images exist in images/ directory

5. **Training Too Slow**
   - Ensure CUDA is available and working
   - Reduce `num_workers` in data loaders if CPU is bottleneck
   - Consider using a smaller batch size

## 📈 Expected Results

After training, you should see:
- Training and validation error curves
- Sample images with predicted vs actual nose locations
- Error distribution histogram for test set
- Mean pixel error typically 15-40 pixels for good models

## 📚 References

- PyTorch Documentation: https://pytorch.org/docs/
- torchvision Transforms: https://pytorch.org/vision/stable/transforms.html
- CNN Architectures: Deep Learning by Ian Goodfellow

## 👨‍💻 Author

**Alistair Barfoot**  
Queen's University - ELEC 475  
Fall 2025

---

For questions or issues, please refer to the lab documentation or contact the course instructors.