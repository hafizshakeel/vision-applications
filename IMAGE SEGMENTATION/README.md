# PyTorch implementation of UNet


This is a PyTorch implementation of U-Net, a convolutional network designed for the semantic segmentation of medical images and other image segmentation tasks.

## Features

- Implementation of U-Net in PyTorch
- Data augmentation with Albumentations
- Support for mixed precision training with AMP
- Save and load model checkpoints
- Save segmentation predictions as image files
- Well-documented code for easier understanding

## Requirements

To run the code, you need to have the following Python packages installed. You can install them using `pip`:

```bash
pip install -r requirements.txt
```


## Usage

### Configuration
Edit `config.py` to set paths and hyperparameters. You can also pass configuration options as command-line arguments.

### Training
To start training the U-Net model, run the `train.py` script with the desired configuration:

```bash
python unet_train.py --train_data_path data/train_images/ --train_ori_data_path data/train_masks/ --val_data_path data/val_images/ --val_ori_data_path data/val_masks/ --test_data_path data/test_images/ --test_ori_data_path data/test_masks_images/ --batch_size 2 --epochs 100
```


### Testing

To test the model and save predictions, ensure that your model is trained and use the `test.py` script with the appropriate command-line arguments:

```bash
python test.py --test_single_img_data_path data/test_single_images/ --test_single_img_ori_data_path data/test_single_mask_images/ --sample_test_output_folder test_images/
```

### Dataset
This implementation assumes you have the Carvana dataset, which includes image files and corresponding mask files. Ensure that the directory structure matches the paths specified in `config.py`.
You can download the dataset from [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge/data)


### Acknowledgements
U-Net: Convolutional Networks for Biomedical Image Segmentation [Link](https://arxiv.org/abs/1505.04597)


