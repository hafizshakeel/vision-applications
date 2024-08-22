import sys

import torch
import os
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import transform

"""Yolo Dataset"""
class VOCDataset(Dataset):
    def __init__(self, img_dir, label_dir, csv_file, S=7, B=2, C=20, transform=transform):
        super(VOCDataset, self).__init__()
        self.img_dir = img_dir  # Directory containing image files
        self.label_dir = label_dir  # Directory containing label files
        self.annotations = pd.read_csv(csv_file)  # CSV with image paths and label paths
        self.transform = transform  # Optional transformation function for image and labels
        self.S = S  # Grid size (SxS)
        self.B = B  # Number of bounding boxes per grid cell
        self.C = C  # Number of classes

    def __len__(self):
        return len(self.annotations)  # Number of samples in the dataset

    def __getitem__(self, index):
        # Get the path to the label file
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []

        # Read and parse the bounding boxes from the label file
        with open(label_path) as f:
            for label in f.readlines():
                # Convert class label to int and bounding box coordinates to float
                class_label, x, y, w, h = [
                    float(x) if float(x) != int(float(x)) else int(x)  # Convert to float or int as needed
                    for x in label.replace("\n", "").split()
                ]
                boxes.append([class_label, x, y, w, h])

        # Load the image
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)  # Convert bounding boxes to tensor

        if self.transform:
            # image = self.transform(image)
            image, boxes = self.transform(image, boxes)

        # Initialize the label matrix as a 3D tensor
        label = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        # Debugging: Print label shape after initialization
        # print(f"Label shape after initialization: {label.shape}")

        # Convert bounding boxes to grid cell format
        for box in boxes:
            class_label, x, y, w, h = box.tolist()  # Convert tensor back to list
            class_label = int(class_label)

            # Determine the grid cell (i, j) where the bounding box is located
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i  # Relative position within the cell
            width_cell, height_cell = w * self.S, h * self.S  # Size relative to the cell

            # Ensure indices are within bounds
            if 0 <= i < self.S and 0 <= j < self.S:
                # Debugging: Print the current cell and class
                # print(f"Processing bounding box for cell ({i}, {j}) with class {class_label}")

                # If no object has been assigned to this cell, mark it as occupied
                if label[i, j, 20] == 0:
                    label[i, j, 20] = 1  # Indicate object presence

                # Set bounding box coordinates in the label matrix
                box_coordinate = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label[i, j, 21:25] = box_coordinate

                # One-hot encode the class label in the label matrix
                label[i, j, class_label] = 1
            else:
                print(f"Warning: Bounding box {box} falls outside grid boundaries ({i}, {j})")

        return image, label


def test_voc_dataset(img_dir, label_dir, csv_file):
    dataset = VOCDataset(img_dir, label_dir, csv_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Sample {batch_idx}:")
        print(f"Image shape: {images.shape}")  # Print shape of image tensor
        print(f"Label shape: {labels.shape}")  # Print shape of label tensor
        if batch_idx >= 2:  # Limit the number of samples for debugging
            break

# Example usage
img_dir = 'data/train_images/'
label_dir = 'data/train_labels/'
csv_file = 'data/annotations.csv'
# test_voc_dataset(img_dir, label_dir, csv_file)


# def test_dataset_shapes(dataset, num_samples=5):
#     for i in range(num_samples):
#         image, label = dataset[i]
#
#         # Print the shapes
#         print(f"Sample {i}:")
#         print(f"Image shape: {image.size} (Height, Width)")
#         print(f"Label shape: {label.shape} (Grid Size, Grid Size, Channels)")
#         sys.exit()
#
#         # Optional: Visualize the image and label if needed
#         # plt.figure(figsize=(12, 6))
#         # plt.subplot(1, 2, 1)
#         # plt.title(f"Sample {i} Image")
#         # plt.imshow(image)
#         # plt.axis('off')
#         #
#         # plt.subplot(1, 2, 2)
#         # plt.title(f"Sample {i} Label")
#         # # Visualize label as a heatmap or similar
#         # plt.imshow(label[:, :, 20].numpy(), cmap='hot', interpolation='nearest')
#         # plt.axis('off')
#         #
#         # plt.show()
#
#
# # Example usage
# if __name__ == "__main__":
#     # Define your dataset with appropriate paths and parameters
#     dataset = VOCDataset(
#         img_dir='data/train_images',
#         label_dir='data/train_labels',
#         csv_file='data/annotations.csv',
#         S=7, B=2, C=20,
#         transform=None  # Apply transformations if needed
#     )
#
#     test_dataset_shapes(dataset)
