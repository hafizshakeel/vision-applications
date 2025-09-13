# Vision Applications

This repository contains PyTorch implementations of popular computer vision models and architectures. Each implementation is organized in its own directory with complete documentation and training scripts.

## Implementations

### 1. Image Segmentation
- Implementation of U-Net architecture for semantic segmentation
- Located in [`IMAGE SEGMENTATION`](./IMAGE%20SEGMENTATION/)
- Includes training, validation, and testing pipelines
- Optimized for the Carvana Image Masking Challenge dataset

### 2. Object Detection
- Implementation of YOLOv1 for real-time object detection
- Located in [`OBJECT DETECTION/YOLOv1`](./OBJECT%20DETECTION/YOLOv1/)
- Complete training pipeline with support for custom datasets
- Includes evaluation metrics (IoU, mAP) and visualization tools

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/hafizshakeel/vision-applications.git
cd vision-applications
```

2. Navigate to the specific implementation directory:
```bash
cd "IMAGE SEGMENTATION"  # For U-Net implementation
# or
cd "OBJECT DETECTION/YOLOv1"  # For YOLO implementation
```

3. Install the required dependencies for the specific implementation:
```bash
pip install -r requirements.txt
```

4. Follow the README in each directory for detailed usage instructions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Contact
  
Email: hafizshakeel1997@gmail.com