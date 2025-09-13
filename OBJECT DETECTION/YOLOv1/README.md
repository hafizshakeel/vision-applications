# YOLOv1 Object Detection Implementation

This directory contains a PyTorch implementation of the YOLOv1 (You Only Look Once) architecture for real-time object detection, as described in the paper ["You Only Look Once: Unified, Real-Time Object Detection"](https://arxiv.org/abs/1506.02640).

## Project Structure

- `model.py`: YOLOv1 architecture implementation
- `dataset.py`: VOC dataset loader with grid cell conversion
- `loss.py`: YOLOv1 loss function implementation
- `metrics.py`: Evaluation metrics (IoU, mAP, NMS)
- `utils.py`: Utility functions for training and visualization
- `train.py`: Training script with optimization settings
- `config.py`: Configuration and hyperparameters
- `data/`: Scripts for dataset preparation

## Features

- Complete YOLOv1 architecture implementation
- Object detection with grid-based predictions
- Support for multiple bounding box predictions per cell
- Evaluation metrics:
  - Intersection over Union (IoU)
  - Mean Average Precision (mAP)
  - Non-Maximum Suppression (NMS)
- Training optimizations:
  - Batch normalization
  - Leaky ReLU activations
  - Dropout regularization
  - Adam optimizer with weight decay

## Dataset

The implementation is designed for the PASCAL VOC dataset. Use the provided script to prepare the dataset:

1. Navigate to the data directory:
```bash
cd data
```

2. Run the dataset preparation script:
```bash
bash get_data
```

This will:
- Download VOC2007 and VOC2012 datasets
- Convert annotations to YOLO format
- Create training and testing splits
- Generate CSV files for data loading

## Model Architecture

- Darknet-inspired architecture
- 24 convolutional layers
- 2 fully connected layers
- Output shape: S×S×(B*5 + C)
  - S: Grid size (7×7)
  - B: Bounding boxes per cell (2)
  - C: Number of classes (20)

## Usage

### Training

1. Configure parameters in `config.py` or pass as command-line arguments:

2. Start training:
```bash
python train.py \
    --train_data_path data/train_images/ \
    --train_label_data_path data/train_labels/ \
    --train_csv_annotations data/annotations.csv \
    --batch_size 2 \
    --epochs 100
```

## Implementation Details

1. **Model Components**:
   - Convolutional backbone based on Darknet
   - Grid-based prediction system
   - Class probability predictions
   - Bounding box coordinates and confidence scores

2. **Loss Function**:
   - Coordinate loss with increased weight (λcoord = 5)
   - Object confidence loss
   - No-object confidence loss with decreased weight (λnoobj = 0.5)
   - Class probability loss
   - Unified end-to-end training

3. **Training Process**:
   - Grid cell assignment based on center points
   - Multiple bounding box predictions per cell
   - Best box selection using highest IoU
   - Class probability conditioning on object existence

4. **Evaluation**:
   - Non-maximum suppression for duplicate removal
   - Mean Average Precision calculation
   - IoU thresholding for true/false positive determination
