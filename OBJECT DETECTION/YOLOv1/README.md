# PyTorch Implementation of YOLOv1

This is a PyTorch implementation of the YOLOv1 (You Only Look Once) model, designed for real-time object detection tasks.


## Features

This implementation of the YOLOv1 model in PyTorch supports custom datasets, data augmentation, and includes comprehensive training and evaluation scripts.
It features evaluation metrics such as Intersection over Union (IoU) and Mean Average Precision (mAP), along with Non-Max Suppression (NMS) for refining predictions. 
The project also includes a dataset testing script, YOLO loss function implementation, and visualization tools for bounding box predictions, all optimized for GPU acceleration.


## Requirements

To run the code, you need to have the following Python packages installed. You can install them using `pip`:

```bash
pip install -r requirements.txt
```


## Usage

### Configuration
Edit `config.py` to set paths, hyperparameters, and other configurations. You can also pass configuration options as command-line arguments.

### Dataset
This implementation assumes that you have a custom dataset formatted with images and corresponding label files. Additional details about the dataset are available inside the `data` folder.

### Training
To start training the YOLOv1 model, run the `train.py` script with the desired configuration:

```bash
python train.py --train_data_path data/train_images/ --train_label_data_path data/train_labels/ --train_csv_annotations data/annotations.csv --batch_size 2 --epochs 100
```

### Acknowledgements
YOLOv1: You Only Look Once: Unified, Real-Time Object Detection [Link](https://arxiv.org/abs/1506.02640)
