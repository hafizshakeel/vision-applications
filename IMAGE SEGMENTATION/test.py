import torch
from torch.utils.data import DataLoader
from unet_model import UNet
from config import get_config
from utils import load_checkpoint, save_predictions_as_imgs
from dataset import CarvanaDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_test_loaders(cfg):
    test_transform = A.Compose([
        A.Resize(height=cfg.img_height, width=cfg.img_width),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    test_ds = CarvanaDataset(image_dir=cfg.test_single_img_data_path, mask_dir=cfg.test_single_img_ori_data_path, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, shuffle=False)

    return test_loader


def test(cfg):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU if available, otherwise CPU
    model = UNet(in_channels=cfg.in_channels, out_channels=cfg.out_channels).to(device=DEVICE)  # Initialize the model

    # Load the saved model checkpoint
    checkpoint = torch.load("model_checkpoint.pth.tar", map_location=DEVICE)
    load_checkpoint(checkpoint, model)

    test_loader = get_test_loaders(cfg)  # Load the test data loader
    save_predictions_as_imgs(test_loader, model, folder=cfg.sample_test_output_folder, device=DEVICE)  # save images


if __name__ == "__main__":
    config_args, unparsed_args = get_config()  # Get configuration settings
    test(config_args)  # Run the test function

