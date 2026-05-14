import torch
from torch import nn
import torch.nn.functional as F
from ..utils import wrap_module_with_gradient_checkpointing
from .encoder import LayerNorm



class DoubleConv(nn.Module):
    """(convolution => [LN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm(mid_channels, eps=1e-6, data_format="channels_first"),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm(out_channels, eps=1e-6, data_format="channels_first"),
            nn.ReLU(inplace=True)
        )

    def enable_gradient_checkpointing(self):
        wrap_module_with_gradient_checkpointing(self.double_conv)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def enable_gradient_checkpointing(self):
        wrap_module_with_gradient_checkpointing(self.maxpool_conv)

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, is_last=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            if is_last:
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=4)
            else:
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def enable_gradient_checkpointing(self):
        wrap_module_with_gradient_checkpointing(self.up)
        wrap_module_with_gradient_checkpointing(self.conv)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UnetDecoder(nn.Module):
    def __init__(self, out_channels, dims, in_channels=3, bilinear=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.up1 = Up(dims[3], dims[2], bilinear)
        self.up2 = Up(dims[2], dims[1], bilinear)
        self.up3 = Up(dims[1], dims[0], bilinear)
        self.up4 = Up(dims[0], self.in_channels, bilinear)
        self.outc = OutConv(self.in_channels, self.out_channels)

    def enable_gradient_checkpointing(self):
        self.up1.enable_gradient_checkpointing()
        self.up2.enable_gradient_checkpointing()
        self.up3.enable_gradient_checkpointing()
        self.up4.enable_gradient_checkpointing()
        wrap_module_with_gradient_checkpointing(self.outc.conv)

    def forward(self, x, features):
        x = self.up1(x, features[3])
        x = self.up2(x, features[2])
        x = self.up3(x, features[1])
        x = self.up4(x, features[0])
        x = self.outc(x)
        return x
