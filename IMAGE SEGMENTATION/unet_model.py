"""
Convolutional Networks for Biomedical Image Segmentation
(U-Net) Implementation: https://arxiv.org/abs/1505.04597

This script contains the implementation of the U-Net model.

Implementation by: Hafiz Shakeel Ahmad Awan
Email: hafizshakeel1997@gmail.com
"""

# Import necessary libraries
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


"""UNET Model"""
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # down-part of unet
        for feature in features:
            self.downs.append(DoubleConv(in_channels=in_channels, out_channels=feature))
            in_channels = feature

        # up-part of unet
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, 2, 2))  # 512X2=1024, 512
            self.ups.append(DoubleConv(feature * 2, feature))  # 512X2=1024, 512

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)  # 512, 1024
        self.last_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)  # 64, 1

    def forward(self, x):
        skip_connections = []  # list of skip_conn from high resolution to low res

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # now reverse the list for up part -- low res to high-resolution

        for idx in range(0, len(self.ups), 2):  # two self.ups --> up, DoubleConv ...
            x = self.ups[idx](x)  # ConvTranspose2d in self.ups
            skip_connection = skip_connections[idx // 2]  # since we've step of 2 in for idx...; and
            # obviously we want skip_coon in a linear step of ordering

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection[2:])  # (-, -, H, W) Note: Authors used cropped in paper

            concat_skip = torch.cat((skip_connection, x), dim=1)  # concat along channel dimension (B, C, H, W)
            x = self.ups[idx + 1](concat_skip)

        return self.last_conv(x)


# def test():
#     x = torch.randn(1, 1, 160, 160)
#     model = UNet(in_channels=1, out_channels=1)
#     preds = model(x)
#     print(preds.shape)
#     assert preds.shape == x.shape
#
#
# if __name__ == '__main__':
#     test()
