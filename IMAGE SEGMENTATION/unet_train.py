""" Training Script for UNet Model """

import torch
import torch.nn as nn
import torch.optim as optim
from unet_model import UNet
from config import get_config
from utils import *  # Utility functions


def main(cfg):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU if available, otherwise CPU
    model = UNet(in_channels=cfg.in_channels, out_channels=cfg.out_channels).to(DEVICE)  # Initialize the model

    # Use CrossEntropyLoss for multiple classes, otherwise BCEWithLogitsLoss
    loss_fn = nn.CrossEntropyLoss() if cfg.num_classes > 1 else nn.BCEWithLogitsLoss()
    optimizer = load_optimizer(model, cfg)

    # Load the data loaders for training and validation, including data augmentation using albumentations library
    train_loader, val_loader = get_loaders(cfg)


    # Optionally load a pre-trained model
    if cfg.load_model:
        load_checkpoint(torch.load("model_checkpoint.pth.tar"), model)

    # Set up gradient scaling for mixed precision training if on GPU
    # For faster training --> https://pytorch.org/docs/stable/amp.html
    scaler = torch.cuda.amp.GradScaler() if DEVICE == "cuda" else None


    print('Start train')

    # Training loop
    for epoch in range(cfg.epochs):
        train_fn(train_loader, model, loss_fn, optimizer, scaler, DEVICE)  # Train for one epoch

        # Save the model's state and optimizer
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint)

        check_accuracy(val_loader, model, DEVICE)  # Check and print model accuracy
        save_predictions_as_imgs(val_loader, model, folder=cfg.sample_output_folder, device=DEVICE)  # Save predictions


if __name__ == '__main__':
    config_args, unparsed_args = get_config()  # Get configuration settings
    main(config_args)  # Begin training with the loaded configuration

