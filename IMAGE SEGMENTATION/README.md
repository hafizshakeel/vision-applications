# U-Net Image Segmentation Implementation

This directory contains a PyTorch implementation of the U-Net architecture for semantic image segmentation, as described in the paper ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/abs/1505.04597).

## Project Structure

- `unet_model.py`: Implementation of the U-Net architecture with skip connections
- `dataset.py`: Custom dataset class for the Carvana Image Masking Challenge
- `utils.py`: Utility functions for data loading, training, and evaluation
- `config.py`: Configuration file with model parameters and dataset paths
- `unet_train.py`: Training script with support for mixed precision training
- `test.py`: Script for evaluating the model and generating predictions

## Features

- **Data Augmentation**: Utilizes Albumentations library for robust augmentation:
  - Random rotation
  - Horizontal and vertical flips
  - Normalization
  - Resizing
- **Training Optimizations**:
  - Mixed precision training with AMP
  - Gradient scaling
  - Batch normalization
  - Configurable learning rate and weight decay
- **Evaluation Metrics**:
  - Pixel-wise accuracy
  - Dice coefficient score
- **Visualization**: Saves prediction masks during training and testing

## Dataset

The implementation is designed for the [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge/data) dataset. Download the dataset and organize it in the following structure:

```
data/
├── train_images/
├── train_masks/
├── val_images/
├── val_masks/
├── test_single_images/
└── test_single_mask_images/
```

## Usage

### Training

1. Configure the dataset paths and hyperparameters in `config.py` or pass them as command-line arguments.

2. Start training:
```bash
python unet_train.py \
    --train_data_path data/train_images/ \
    --train_ori_data_path data/train_masks/ \
    --val_data_path data/val_images/ \
    --val_ori_data_path data/val_masks/ \
    --batch_size 2 \
    --epochs 100
```

Training progress will be displayed with loss values and segmentation metrics. Predictions will be saved in the `saved_images/` directory.

### Testing

Run inference on test images:
```bash
python test.py \
    --test_single_img_data_path data/test_single_images/ \
    --test_single_img_ori_data_path data/test_single_mask_images/ \
    --sample_test_output_folder test_images/
```

Segmentation masks will be saved in the `test_images/` directory.

## Model Architecture

The U-Net implementation follows the original architecture with some modern improvements:
- Double convolutional blocks with batch normalization
- ReLU activation functions
- Skip connections between encoder and decoder
- Input/output size preservation using padding

## Implementation Details

1. **Double Convolution Block**:
   - Two 3x3 convolutions with batch normalization and ReLU
   - Used in both encoder and decoder paths

2. **Encoder Path**:
   - Progressively reduces spatial dimensions
   - Increases feature channels (64 → 128 → 256 → 512)
   - MaxPooling between blocks

3. **Decoder Path**:
   - Transposed convolutions for upsampling
   - Skip connections from encoder
   - Feature channel reduction
   - Final 1x1 convolution for segmentation output

4. **Loss Function**:
   - Binary Cross Entropy with Logits for binary segmentation
   - Cross Entropy Loss for multi-class segmentation


