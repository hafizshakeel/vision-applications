"""
You Only Look Once: Unified, Real-Time Object Detection
(YOLOv1) Implementation: https://arxiv.org/abs/1506.02640

This script contains the implementation of the YOLOv1 model.

Implementation by: Hafiz Shakeel Ahmad Awan
Email: hafizshakeel1997@gmail.com

Task 1. Understanding and implementation of IoU metric, nms, mean_avg_precision
Task 2: Implementation of model, load dataset and training setup

"""

# Import necessary libraries
import torch
import torch.nn as nn

# Architecture configuration:
# Structure: (kernel_size, filters, stride, padding)
# and "M" stands for max-pooling with kernel 2x2 and stride 2

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 92, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        # Define a convolutional block with BatchNorm and LeakyReLU
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)  # Convolutional layer
        self.bn = nn.BatchNorm2d(out_channels)  # Batch normalization
        self.leakyrelu = nn.LeakyReLU()  # Leaky ReLU activation

    def forward(self, x):
        # Forward pass through the convolutional block
        return self.leakyrelu(self.bn(self.conv(x)))


class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config  # Network architecture
        self.in_channels = in_channels  # Number of input channels (e.g., 3 for RGB)
        self.darknet = self._make_layers(self.architecture)  # Build the network layers
        self.fcs = self._create_fcs(**kwargs)  # Define fully connected layers

    def forward(self, x):
        # Forward pass through the network
        x = self.darknet(x)  # Apply convolutional layers
        return self.fcs(torch.flatten(x, start_dim=1))  # Flatten and apply fully connected layers

    def _make_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                # Add a convolutional block
                layers += [CNNBlock(in_channels, out_channels=x[1], kernel_size=x[0], stride=x[2],
                                    padding=x[3])]
                in_channels = x[1]  # Update the number of input channels for the next layer
            elif type(x) == str:
                # Add a max pooling layer
                layers += [nn.MaxPool2d(2, 2)]

            elif type(x) == list:
                # Add repeated convolutional blocks
                conv1 = x[0]  # First convolutional layer in the block
                conv2 = x[1]  # Second convolutional layer in the block
                num_repeat = x[2]  # Number of repetitions

                for _ in range(num_repeat):
                    # Add first convolutional block
                    layers += [CNNBlock(in_channels, out_channels=conv1[1], kernel_size=conv1[0],
                                        stride=conv1[2], padding=conv1[3])]
                    # Add second convolutional block
                    layers += [CNNBlock(in_channels=conv1[1], out_channels=conv2[1], kernel_size=conv2[0],
                                        stride=conv2[2], padding=conv2[3])]

                in_channels = conv2[1]  # Update input channels for subsequent layers

        return nn.Sequential(*layers)  # Combine layers into a Sequential model

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        return nn.Sequential(
            nn.Flatten(),  # Flatten the output from convolutional layers
            nn.Linear(1024 * S * S, 4096),  # Fully connected layer
            nn.Dropout(0.5),  # Dropout for regularization
            nn.LeakyReLU(0.1),  # Leaky ReLU activation
            nn.Linear(4096, S * S * (C + B * 5))  # Output layer with size for predictions
            # Output shape: (S * S * (C + B * 5)), where C + B*5 = 30 for 20 classes and 2 boxes
        )


# Uncomment for testing the model
# def test(S=7, B=2, C=20):
#     model = Yolov1(split_size=S, num_boxes=B, num_classes=C)
#     x = torch.randn(2, 3, 448, 448)  # Example input tensor
#     print(model(x).shape)  # Should output: [2, 1470] (corresponding to S*S*(C+B*5))
#
#
# if __name__ == '__main__':
#     test()
