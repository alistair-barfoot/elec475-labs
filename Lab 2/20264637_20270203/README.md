# SnoutNet: Deep Learning for Pet Nose Detection

A PyTorch-based deep learning project for detecting and localizing pet noses in images using convolutional neural networks. This project implements three different architectures (SnoutNet, AlexNet, VGG16) and supports data augmentation and ensemble methods.

## 📁 Project Structure

```
├── model.py              # Custom SnoutNet architecture definition
├── dataset.py            # Custom dataset class with augmentation support
├── train.py             # Training script with multiple architecture support
├── test.py              # Testing and validation script
├── ensemble.py          # Ensemble model implementation and validation
├── train_noses.txt      # Training dataset annotations
├── test_noses.txt       # Test dataset annotations
├── images-original/     # Dataset images directory
│   └── images/          # Pet images (cats and dogs)
├── models/              # Trained model weights
│   ├── sw_*.pth         # SnoutNet models
│   ├── alex_*.pth       # AlexNet models
│   └── vgg_*.pth        # VGG16 models
└── SnoutNet*.txt        # Command execution scripts
└── ELEC475_Lab2Report.pdf # Detailed lab report with results and analysis
```

## 🏗️ Architecture

### SnoutNet (Custom Architecture)
- **Input**: 227×227×3 RGB images
- **Architecture**: 3 convolutional layers + 3 fully connected layers
- **Output**: 2D coordinates (x, y) for nose position
- **Features**: 
  - Progressive channel increase (3→64→128→256)
  - Max pooling after each conv layer
  - ReLU activations throughout
  - Final output scaled to image dimensions

### Transfer Learning Models
- **AlexNet**: Pre-trained ImageNet weights, modified final layer for regression
- **VGG16**: Pre-trained ImageNet weights, modified final layer for regression

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch torchvision pillow opencv-python pandas numpy matplotlib torchsummary
```

### Dataset Format
The dataset consists of pet images with nose coordinate annotations:
```
filename.jpg,"(x_coordinate, y_coordinate)"
```

## 🎯 Training

### Basic Training (SnoutNet)
```bash
python train.py -s models/snoutnet_weights.pth -e 50 -b 32
```

### Training with Data Augmentation
```bash
# With random flip augmentation
python train.py -f -s models/snoutnet_flip.pth -e 50 -b 32

# With rotation augmentation  
python train.py -r -s models/snoutnet_rot.pth -e 50 -b 32

# With both augmentations
python train.py -f -r -s models/snoutnet_both.pth -e 50 -b 32
```

### Training Different Architectures
```bash
# AlexNet
python train.py -a -s models/alex_weights.pth -e 50 -b 32

# VGG16
python train.py -v -s models/vgg_weights.pth -e 50 -b 32
```

### Training Parameters
- `-s`: Save path for model weights
- `-e`: Number of epochs (default: 30)
- `-b`: Batch size (default: 32)
- `-p`: Plot file for loss curves
- `-a`: Use AlexNet architecture
- `-v`: Use VGG16 architecture
- `-f`: Enable random flip augmentation
- `-r`: Enable rotation augmentation

## 🧪 Testing

### Single Model Testing
```bash
python test.py -m models/snoutnet_weights.pth -t snout
```

### Testing with Visualization
```bash
# Show predictions with ground truth
python test.py -m models/snoutnet_weights.pth -g -s -v -t snout
```

### Test Parameters
- `-m`: Path to model weights
- `-t`: Model type (snout/alex/vgg)
- `-g`: Show ground truth coordinates
- `-s`: Show individual predictions
- `-v`: Verbose output
- `-a`: Process all test images

## 🎭 Ensemble Methods

### Run Ensemble Prediction
```bash
python ensemble.py -x models/sw_BOTH_AUG.pth -y models/alex_BOTH_AUG.pth -z models/vgg_BOTH_AUG.pth
```

### Ensemble with Visualization
```bash
python ensemble.py -x models/sw_BOTH_AUG.pth -y models/alex_BOTH_AUG.pth -z models/vgg_BOTH_AUG.pth -s -g -v
```

### Ensemble Parameters
- `-x`, `-y`, `-z`: Paths to three different model weights
- `-s`: Show individual predictions
- `-g`: Show ground truth coordinates
- `-v`: Verbose output

## 📊 Model Performance

The project evaluates models using:
- **Mean pixel distance** between predicted and actual nose coordinates
- **Standard deviation** of prediction errors
- **Best/Worst case analysis** (top 4 and bottom 4 predictions)

Performance is reported for:
- Overall dataset statistics
- Best 4 performing images
- Worst 4 performing images

## 🔧 Data Augmentation

Supported augmentation techniques:
- **Random Horizontal Flip**: Mirrors images horizontally with corresponding coordinate adjustment
- **Random Rotation**: Rotates images with proper coordinate transformation
- **Combined Augmentation**: Uses both flip and rotation

The `CustomDataset` class handles coordinate transformations automatically to maintain accuracy during augmentation.

## 📈 Key Features

1. **Flexible Architecture Support**: Switch between SnoutNet, AlexNet, and VGG16
2. **Advanced Data Augmentation**: Coordinate-aware transformations
3. **Ensemble Learning**: Combine multiple models for improved accuracy
4. **Comprehensive Evaluation**: Detailed statistics and visualization
5. **Early Stopping**: Prevents overfitting with patience-based stopping
6. **Learning Rate Scheduling**: Adaptive learning rate reduction

## 🛠️ Implementation Details

### Custom Dataset Class
- Handles coordinate parsing from annotation files
- Implements proper coordinate scaling for different image sizes
- Supports torchvision v2 transformations with bounding box handling

### Training Features
- **Loss Function**: Mean Squared Error (MSE) for coordinate regression
- **Optimizer**: Adam with weight decay
- **Scheduler**: ReduceLROnPlateau for adaptive learning rates
- **Early Stopping**: Monitors validation loss with configurable patience

### Model Loading
- Robust checkpoint loading supporting multiple formats
- Automatic architecture detection from filename
- Flexible state dict handling for different checkpoint structures

## 📝 Usage Examples

### Complete Training Pipeline
```bash
# Train SnoutNet with augmentation
python train.py -f -r -s models/snoutnet_final.pth -p loss_plot.png -e 50 -b 32

# Test the trained model
python test.py -m models/snoutnet_final.pth -g -v -t snout

# Create ensemble with multiple models
python ensemble.py -x models/snoutnet_final.pth -y models/alex_final.pth -z models/vgg_final.pth -v
```

### Batch Processing Scripts
The project includes pre-configured command files:
- `SnoutNet.txt`: SnoutNet training and testing commands
- `SnoutNet-A.txt`: AlexNet training and testing commands  
- `SnoutNet-V.txt`: VGG16 training and testing commands
- `SnoutNet-Ensemble.txt`: Ensemble evaluation commands

## 🎯 Applications

This nose detection system can be used for:
- Pet identification and recognition systems
- Veterinary applications for animal health monitoring
- Animal behavior analysis
- Photo organization and tagging
- Augmented reality pet filters

## 📊 Dataset

The dataset contains images of various dog and cat breeds including:
- **Dogs**: Beagle, Boxer, Chihuahua, German Shorthaired, Great Pyrenees, and more
- **Cats**: Abyssinian, Bengal, Birman, British Shorthair, Maine Coon, and more

Each image is annotated with precise nose coordinates for supervised learning.

## 🏆 Results

The project demonstrates:
- Effective coordinate regression for nose localization
- Benefits of data augmentation on model generalization
- Improved performance through ensemble methods
- Comparison of different CNN architectures for the task

## 📄 Lab Report

The complete analysis and results are documented in `ELEC475_Lab2Report.pdf`, which includes:
- Detailed experimental methodology
- Comprehensive performance comparisons between architectures
- Analysis of data augmentation effects
- Ensemble method evaluation
- Discussion of results and conclusions
- Visual examples of nose detection predictions

---

**Note**: This project was developed as part of ELEC 475 coursework, demonstrating practical applications of deep learning in computer vision tasks.