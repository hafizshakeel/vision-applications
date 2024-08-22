import torch
import torch.nn as nn
from metrics import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        # Parameters:
        # S: Split size of the image grid (7 in the original YOLO paper)
        # B: Number of bounding boxes per cell (2 in the original YOLO paper)
        # C: Number of classes (20 for VOC dataset)
        self.S = S
        self.B = B
        self.C = C

        # Loss weights from YOLO paper
        self.lambda_noobj = 0.5  # Weight for no-object loss
        self.lambda_coord = 5  # Weight for bounding box coordinates loss

    def forward(self, prediction, target):
        # Reshape prediction to match grid size and number of bounding boxes
        prediction = prediction.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Compute IoU for both bounding boxes
        iou_b1 = intersection_over_union(prediction[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(prediction[..., 26:30], target[..., 21:25])
        iou = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # Determine the best bounding box based on IoU
        iou_maxes, best_box = torch.max(iou, dim=0)

        # Check if a box exists in the target (class score > 0)
        exists_box = target[..., 20].unsqueeze(3)

        """ BOX COORDINATE LOSS (x, y, w, h) """
        # Choose the bounding box with the highest IoU for each cell
        box_predictions = exists_box * (best_box * prediction[..., 26:30] + (1 - best_box) * prediction[..., 21:25])
        box_target = exists_box * target[..., 21:25]

        # Apply transformation to bounding box coordinates for numerical stability
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6))
        box_target[..., 2:4] = torch.sqrt(box_target[..., 2:4])

        # Compute MSE loss for bounding box coordinates
        box_loss = self.mse(torch.flatten(box_predictions, end_dim=-2),
                            torch.flatten(box_target, end_dim=-2))

        """ OBJECT LOSS """
        # Compute object confidence score for the best bounding box
        pred_box = (best_box * prediction[..., 25:26] + (1 - best_box) * prediction[..., 20:21])
        object_loss = self.mse(torch.flatten(exists_box * pred_box),
                               torch.flatten(exists_box * target[..., 20:21]))

        """ NO OBJECT LOSS """
        # Compute loss for cells without an object
        no_object_loss = self.mse(torch.flatten((1 - exists_box) * prediction[..., 20:21], start_dim=1),
                                  torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1))
        # Add loss for the second bounding box
        no_object_loss += self.mse(torch.flatten((1 - exists_box) * prediction[..., 25:26], start_dim=1),
                                   torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1))

        """ CLASS LOSS """
        # Compute classification loss for each cell that contains an object
        class_loss = self.mse(torch.flatten(exists_box * prediction[..., :20], end_dim=-2),
                              torch.flatten(exists_box * target[..., :20], end_dim=-2))

        # Total loss combining all components
        loss = (self.lambda_coord * box_loss  # Weight for bounding box coordinates loss
                + object_loss  # Object confidence score loss
                + self.lambda_noobj * no_object_loss  # Weight for no-object loss
                + class_loss)  # Classification loss

        return loss
