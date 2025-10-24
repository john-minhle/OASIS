# OASIS-Net: Adversarial Semi-supervised Network for Cervical and Fetal Ultrasound Imaging

OASIS-Net is a deep learning framework for medical image segmentation using semi-supervised learning with adversarial training techniques. The network combines Gradient Adversarial Perturbation (GAP) with Iterative Fast Gradient Sign Method (I-FGSM) to improve segmentation performance on ultrasound medical imaging.

## 🎯 Overview

This repository implements a semi-supervised semantic segmentation approach specifically designed for cervical and fetal ultrasound imaging. The framework leverages both labeled and unlabeled data to train robust segmentation models while addressing the challenge of limited labeled medical data.

### Key Features

- **Adversarial Training**: Combines I-FGSM (Iterative Fast Gradient Sign Method) with GAP (Gradient Adversarial Perturbation) for robust pseudo-label generation
- **Semi-Supervised Learning**: Efficiently utilizes both labeled and unlabeled data with confidence-based pseudo-labeling
- **Strong-Weak Augmentation**: Implements sophisticated data augmentation strategies for unlabeled data
- **DeepLabV3Plus Architecture**: Uses DeepLabV3
- **K-Fold Cross Validation**: Supports 5-fold cross-validation for robust model evaluation
- **Comprehensive Metrics**: Tracks Dice Score, IoU, and Hausdorff Distance using MONAI

## 📋 Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Dataset Preparation](#dataset-preparation)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Model Architecture](#model-architecture)
- [Methodology](#methodology)
- [Results](#results)
- [Citation](#citation)

## 🚀 Installation

### Requirements

- Python 3.8+
- CUDA 11.x or higher (for GPU support)
- PyTorch 1.10+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/john-minhle/OASIS.git
cd OASIS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Key Dependencies

- PyTorch & torchvision
- MONAI (Medical Open Network for AI)
- Albumentations (data augmentation)
- OpenCV
- NumPy, Pandas
- PyYAML
- tqdm

## 📁 Project Structure

```
OASIS/
├── OASIS.py                    # Main training script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── Configs/
│   └── multi_train_local.yml   # Training configuration
├── Datasets/
│   ├── create_dataset.py       # Dataset creation and loading
│   ├── transform.py            # Data transformations
│   ├── data_augmentation.py    # Augmentation strategies
│   └── unimatch_utils.py       # Utility functions
├── Models/
│   ├── decoder.py              # Decoder architecture
│   └── DeepLabV3Plus/          # DeepLabV3+ implementation
│       ├── modeling.py         # Model definitions
│       ├── _deeplab.py         # DeepLab components
│       ├── utils.py            # Model utilities
│       └── backbone/           # Backbone networks
└── Utils/
    ├── utils.py                # General utilities
    └── thresh_helper.py        # Threshold helpers
```

## 📊 Dataset Preparation

### Data Format

The dataset should be organized as follows:

```
data_processed/
├── fhps/                       # Dataset name
│   ├── images/                 # Input images (.npy format)
│   │   ├── image_001.npy
│   │   ├── image_002.npy
│   │   └── ...
│   ├── labels/                 # Ground truth labels (.npy format)
│   │   ├── image_001.npy
│   │   ├── image_002.npy
│   │   └── ...
│   ├── fold_label_1.txt        # Labeled data for fold 1
│   ├── fold_label_2.txt        # Labeled data for fold 2
│   ├── fold_label_3.txt        # Labeled data for fold 3
│   ├── fold_label_4.txt        # Labeled data for fold 4
│   ├── fold_label_5.txt        # Labeled data for fold 5
│   └── unlabeled.txt           # Unlabeled data filenames
```

### Data Format Specifications

- **Images**: NumPy arrays (.npy) with shape `(H, W, 3)` or `(H, W)` for grayscale
- **Labels**: NumPy arrays (.npy) with shape `(H, W)` containing class indices (0, 1, 2, ...)
- **Text Files**: Plain text files with one filename per line (e.g., `image_001.npy`)

### Key Parameters

- **conf_thresh**: Confidence threshold for accepting pseudo-labels (default: 0.95)
- **img_size**: Input image size (will be resized to this dimension)
- **l_batchsize**: Batch size for labeled data
- **u_batchsize**: Batch size for unlabeled data
- **num_epochs**: Total training epochs
- **num_classes**: Number of segmentation classes (background + foreground classes)

## 🏋️ Training

### Basic Training

Train the model with default settings:

```bash
python OASIS.py --config_yml Configs/multi_train_local.yml --exp experiment_name
```

### Training Process

The training script will:
1. Load labeled and unlabeled data based on fold splits
2. Initialize DeepLabV3+ model with pretrained ResNet101 backbone
3. Train using combined supervised and unsupervised losses
4. Validate after each epoch
5. Save the best model based on validation Dice score
6. Generate logs and results in the experiment directory

### Output Structure

```
checkpoints/fhps_test/my_experiment/
├── fold1/
│   ├── best.pth                # Best model weights
│   ├── exp_config.yml          # Configuration used
│   ├── log.txt                 # Training logs
│   └── test_results.txt        # Test metrics
├── fold2/
│   └── ...
└── fold5/
    └── ...
```

## 📈 Evaluation

The model is automatically evaluated during training. Test results include:

- **Dice Score**: Overlap between prediction and ground truth
- **IoU (Intersection over Union)**: Jaccard index
- **Hausdorff Distance**: Maximum distance between boundaries
- **Loss**: Validation loss

Results are saved in `test_results.txt` for each fold.
