""" Script for training Yolov1 model on Pascal VOC dataset """

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import get_config
from dataset import VOCDataset
from loss import YoloLoss
from metrics import mean_average_precision
from model import Yolov1
from utils import *


def main(cfg):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Select device: GPU if available, otherwise CPU
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)  # Initialize YOLOv1 model
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )  # Adam optimizer with learning rate and weight decay
    loss_fn = YoloLoss()  # Define loss function

    # Uncomment to load a pre-trained model if needed
    # if cfg.load_model:
    #     load_checkpoint(torch.load(cfg.load_model_file), model, optimizer)

    # Train on a small batch initially for testing; switch to full dataset later
    train_dataset = VOCDataset(img_dir=cfg.train_data_path, label_dir=cfg.train_label_data_path,
                               csv_file=cfg.train_csv_annotations, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                              pin_memory=cfg.pin_memory, shuffle=True, drop_last=False)
    # drop_last=False ensures all data is used, but can be set to True to avoid small batch issues

    for epoch in range(cfg.num_epochs):
        # Uncomment for debugging: visualize predictions and ground truth
        # for x, y in train_loader:
        #    x = x.to(DEVICE)
        #    for idx in range(10):  # Visualize predictions for first 10 examples
        #        bboxes = cellboxes_to_boxes(model(x))
        #        bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
        #        plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)
        #    import sys
        #    sys.exit()

        # Compute and print mean Average Precision (mAP) for current epoch
        pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4)
        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
        # print(f"Train mAP: {mean_avg_prec}")

        # Uncomment to save model checkpoint if performance is satisfactory
        # if mean_avg_prec > 0.9:
        #     checkpoint = {
        #         "state_dict": model.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #     }
        #     save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
        #     import time
        #     time.sleep(10)

        train_fn(train_loader, model, optimizer, loss_fn, DEVICE)  # Train the model for one epoch


if __name__ == '__main__':
    config_args, unparsed_args = get_config()  # Get configuration settings
    main(config_args)  # Begin training with the loaded configuration


# Some Notes:
# Consider using a pre-trained model for better performance.
# Data augmentation and pre-training like ResNet or ViT are important for improving results.
# Resize the input images to match pre-trained models like ResNet or ViT if you choose to use them.
