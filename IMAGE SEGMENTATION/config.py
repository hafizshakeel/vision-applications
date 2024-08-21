import argparse  # Import necessary libraries
parser = argparse.ArgumentParser()  # Initialize the argument parser

"""
Paths for the training, validation, and test datasets.
Use './' or '../' to navigate to data folders within or outside the directory.
"""

parser.add_argument('--train_data_path', type=str, default='data/train_images/', help='Train image path')
parser.add_argument('--train_ori_data_path', type=str, default='data/train_masks/', help='GT image path')
parser.add_argument('--val_data_path', type=str, default='data/val_images/', help='Validation image path')
parser.add_argument('--val_ori_data_path', type=str, default='data/val_masks/', help='Validation GT image path')

parser.add_argument('--test_data_path', type=str, default='data/test_images/', help='Test image path')
parser.add_argument('--test_ori_data_path', type=str, default='data/test_masks_images/', help='Test GT image path')
parser.add_argument('--test_single_img_data_path', type=str, default='data/test_single_images/', help='Test image path')
parser.add_argument('--test_single_img_ori_data_path', type=str, default='data/test_single_mask_images/', help='Test GT image path')
parser.add_argument('--sample_output_folder', type=str, default='saved_images/', help='Sample output folder path')
parser.add_argument('--sample_test_output_folder', type=str, default='test_images/', help='Sample output folder path')

"""Model Parameters and Hyperparameters"""

parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--num_workers', type=int, default=0, help='Number of threads for data loader, for window set to 0')
parser.add_argument('--batch_size', type=int, default=2, help='Training batch size')
parser.add_argument('--val_batch_size', type=int, default=2, help='Validation batch size')
parser.add_argument('--epochs', type=int, default=2, help='Number of epochs for training')
parser.add_argument('--model_dir', type=str, default='./models/', help='Directory where models are saved')
parser.add_argument('--log_dir', type=str, default='./log', help='Directory for saving logs')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='L2 regularization for the optimizer')
parser.add_argument('--net_name', type=str, default='model_', help='Name prefix for the saved models')
parser.add_argument('--grad_clip_norm', type=float, default=0.1, help='Prevent exploding gradients')
parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels')
parser.add_argument('--out_channels', type=int, default=1, help='Number of output channels')
parser.add_argument('--num_classes', type=int, default=10, help='Number of output classes')
# parser.add_argument('--load_model', type=bool, default=False, help='Continue training from the last epoch')
parser.add_argument('--load_model', action='store_true', help='Continue training from the last epoch')
parser.add_argument('--img_width', type=int, default=240, help='Width of an image')
parser.add_argument('--img_height', type=int, default=160, help='Height of an image')
parser.add_argument('--pin_memory', type=bool, default=True, help='Faster data transfer to CUDA-enabled GPU via pin memory')


# Function to get the configuration and unparsed arguments
def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
