import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


def get_loaders(cfg):
    # data augmentation using albumentations library
    train_transform = A.Compose(
        [
            A.Resize(height=cfg.img_height, width=cfg.img_width),  # Resize images
            A.Rotate(limit=35, p=1.0),  # Random rotation
            A.HorizontalFlip(p=0.5),  # Random horizontal flip
            A.VerticalFlip(p=0.1),  # Random vertical flip
            A.Normalize(
                mean=[0.0, 0.0, 0.0],  # Normalize images
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),  # Convert to PyTorch tensor
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height=cfg.img_height, width=cfg.img_width),  # Resize images
            A.Normalize(
                mean=[0.0, 0.0, 0.0],  # Normalize images
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),  # Convert to PyTorch tensor
        ]
    )

    train_ds = CarvanaDataset(image_dir=cfg.train_data_path, mask_dir=cfg.train_ori_data_path, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                              shuffle=True)

    val_ds = CarvanaDataset(image_dir=cfg.val_data_path, mask_dir=cfg.val_ori_data_path, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                            shuffle=False)

    return train_loader, val_loader


def train_fn(loader, model, loss_fn, optimizer, scaler, device):
    loop = tqdm(loader)

    for idx, (data, target) in enumerate(loop):
        data = data.to(device=device)
        target = target.float().unsqueeze(1).to(device=device)  # adding channel in masks
        # target = target.float().unsqueeze(1)

        # Only use mixed precision if scaler is not None (i.e., CUDA is available)
        # Forward, Backward propagation
        if scaler:
            with torch.cuda.amp.autocast():
                preds = model(data)
                loss = loss_fn(preds, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(data)
            loss = loss_fn(preds, target)
            loss.backward()
            optimizer.step()

        # update tqdm loop
        loop.set_postfix(loss=loss.item)


def save_checkpoint(state, filename="model_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def load_optimizer(net, cfg):
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return optimizer


def check_accuracy(loader, model, device):
    num_correct = 0
    num_pixels = 0  # as in segmentation, we predict output / class for each individual pixel
    dice_score = 0  # metric for image segmentation other than accuracy
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device).unsqueeze(1)  # channel dimension

            pred = torch.sigmoid(model(x))  # sigmoid since one class predicton
            pred = (pred > 0.5).float()  # for binary - thresholding

            num_correct += (pred == y).sum()
            num_pixels += torch.numel(pred)

            # Important to  read!
            # In binary segmentation, using accuracy as a performance metric can be misleading due to class imbalance,
            # as it only measures the percentage of correctly classified pixels without considering the distribution
            # of foreground and background. For example, predicting all pixels as background in an image where >80%
            # of the pixels are background can still give >80% accuracy but fails to identify the foreground.
            # A better metric is the Dice coefficient, which measures the overlap between the predicted and actual
            # foreground pixels, providing a more balanced and informative evaluation. The Dice coefficient ranges
            # from 0 (no overlap) to 1 (perfect overlap), making it especially useful in fields requiring precise
            # boundary detection, like medical image segmentation.

            # DICE Score = (2 * Intersection) / (Area of Set A + Area of Set B) or Total Area |2 * X ∩ Y| / |x|+|y|
            # (pred * y).sum() --> Elementwise multipy where both outputting a white pixel  and sum all those
            # (pred + y).sum() --> No of pixels outputting 1 for both of them and take sum
            # Here dice score is  for binary - can be modified for multiple class segmentation

            dice_score += (2 * (pred * y).sum()) / ((pred + y).sum() + 1e-8)

        print(f"Got {num_correct}/{num_pixels} with accuracy {num_correct / num_pixels * 100:.2f}")
        print(f"Dice score: {dice_score}/{len(loader)}")
        model.train()


def save_predictions_as_imgs(loader, model, folder, device):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")  # prediction
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")  # corresponding correct to prediction
    model.train()



