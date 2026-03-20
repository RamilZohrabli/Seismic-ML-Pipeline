import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # handle odd sizes safely
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        x = F.pad(
            x,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class TinyUNet(nn.Module):
    """
    Lightweight U-Net for CPU / low-memory training.
    Input : (B, 1, 751, 256)
    Output: (B, 1, 751, 256)
    """
    def __init__(self, in_channels=1, out_channels=1, base_ch=8):
        super().__init__()

        self.inc = DoubleConv(in_channels, base_ch)          # 8
        self.down1 = Down(base_ch, base_ch * 2)              # 16
        self.down2 = Down(base_ch * 2, base_ch * 4)          # 32
        self.down3 = Down(base_ch * 4, base_ch * 8)          # 64

        self.up1 = Up(base_ch * 8, base_ch * 4, base_ch * 4) # 64+32 -> 32
        self.up2 = Up(base_ch * 4, base_ch * 2, base_ch * 2) # 32+16 -> 16
        self.up3 = Up(base_ch * 2, base_ch, base_ch)         # 16+8  -> 8

        self.outc = nn.Conv2d(base_ch, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)      # (B, 8, 751, 256)
        x2 = self.down1(x1)   # (B, 16, ...)
        x3 = self.down2(x2)   # (B, 32, ...)
        x4 = self.down3(x3)   # (B, 64, ...)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        logits = self.outc(x)
        return logits