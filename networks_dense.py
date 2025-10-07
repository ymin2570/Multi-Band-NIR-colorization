import torch.nn.functional as F
from SENet import *
import math


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, bias=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv_blocks = nn.ModuleList()
        self.relu = nn.ReLU()
        in_temp = in_channels
        out_temp = out_channels // 2  # //4

        for i in range(4):
            self.conv_blocks.append(nn.Conv2d(in_temp, out_temp, kernel_size=3, padding=1, bias=bias))
            in_temp = in_temp + out_temp

        self.out_conv = nn.Conv2d(in_temp, out_channels, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        for block in self.conv_blocks:
            x = torch.cat([x, self.relu(block(x))], 1)
        return self.relu(self.out_conv(x))


class Down(nn.Module):
    """Down-scaling with max pooling then DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Up-scaling then DoubleConv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):  # x2: copied feature maps
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
    """Map to the desired number of channels"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 1x1 convolution

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(UNet, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.inc = DoubleConv(self.in_ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, self.out_ch)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = x

        return x

class UNetKD(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(UNetKD, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.inc = DoubleConv(self.in_ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, self.out_ch)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        x10 = self.outc(x9)

        return x1, x2, x3, x4, x5, x6, x7, x8, x9, x10

class UNetKDPhase(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(UNetKDPhase, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.inc = DoubleConv(self.in_ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.fpre1 = nn.Conv2d(512, 512, 1, 1, 0)
        self.frequency_process1 = FreBlock(512)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.fpre2 = nn.Conv2d(256, 256, 1, 1, 0)
        self.frequency_process2 = FreBlock(256)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.fpre3 = nn.Conv2d(128, 128, 1, 1, 0)
        self.frequency_process3 = FreBlock(128)
        self.up4 = Up(128, 64, bilinear)
        self.fpre4 = nn.Conv2d(64, 64, 1, 1, 0)
        self.frequency_process4 = FreBlock(64)
        self.outc = OutConv(64, self.out_ch)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x6_freq = torch.fft.rfft2(self.fpre1(x6), norm='backward')
        _, pha6 = self.frequency_process1(x6_freq, mode='none')
        x7 = self.up2(x6, x3)
        x7_freq = torch.fft.rfft2(self.fpre2(x7), norm='backward')
        _, pha7 = self.frequency_process2(x7_freq, mode='none')
        x8 = self.up3(x7, x2)
        x8_freq = torch.fft.rfft2(self.fpre3(x8), norm='backward')
        _, pha8 = self.frequency_process3(x8_freq, mode='none')
        x9 = self.up4(x8, x1)
        x9_freq = torch.fft.rfft2(self.fpre4(x9), norm='backward')
        _, pha9 = self.frequency_process4(x9_freq, mode='none')
        x10 = self.outc(x9)

        return pha6, pha7, pha8, pha9, x10


class FreBlock(nn.Module):
    def __init__(self, nc):
        super(FreBlock, self).__init__()
        self.process = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))

    def forward(self, x, mode='none'):
        x = x + 1e-8
        mag = torch.abs(x)
        pha = torch.angle(x)
        if mode == 'amplitude':
            mag = self.process(mag)
        elif mode == 'phase':
            pha = self.process(pha)
        else:
            pass

        return mag, pha


class ColorGuidedStructureModule(nn.Module):
    def __init__(self, n_feat):
        super(ColorGuidedStructureModule, self).__init__()
        self.a = nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0)
        self.b = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0),
        )

    def forward(self, a, b):
        a_out = self.a(a)
        a_res = a_out - a
        pooling = torch.nn.functional.adaptive_avg_pool2d(a_res, (1, 1))
        attn = torch.nn.functional.softmax(pooling, dim=1)
        b1 = b * attn
        b1 = self.b(b1)
        b_out = b1 + b

        return a_out, b_out


class UNetKDPhaseParallel(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(UNetKDPhaseParallel, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.inc = DoubleConv(self.in_ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.fpre1 = nn.Conv2d(512, 512, 1, 1, 0)
        self.frequency_process1 = FreBlock(512)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.fpre2 = nn.Conv2d(256, 256, 1, 1, 0)
        self.frequency_process2 = FreBlock(256)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.fpre3 = nn.Conv2d(128, 128, 1, 1, 0)
        self.frequency_process3 = FreBlock(128)
        self.up4 = Up(128, 64, bilinear)
        self.fpre4 = nn.Conv2d(64, 64, 1, 1, 0)
        self.frequency_process4 = FreBlock(64)

        self.inc_2 = DoubleConv(self.in_ch, 64)
        self.down1_2 = Down(64, 128)
        self.down2_2 = Down(128, 256)
        self.down3_2 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4_2 = Down(512, 1024 // factor)
        self.up1_2 = Up(1024, 512 // factor, bilinear)
        self.up2_2 = Up(512, 256 // factor, bilinear)
        self.up3_2 = Up(256, 128 // factor, bilinear)
        self.up4_2 = Up(128, 64, bilinear)

        # Structure Guided Color Module
        self.CGS1 = ColorGuidedStructureModule(512 // factor)
        self.CGS2 = ColorGuidedStructureModule(256 // factor)
        self.CGS3 = ColorGuidedStructureModule(128 // factor)
        self.CGS4 = ColorGuidedStructureModule(64 // factor)

        self.se = SELayer(channel=64)
        self.outc = OutConv(64, self.out_ch)

    @staticmethod
    def pce(c, f):
        # c: color feature, f: content feature, e: output feature
        m = -torch.mean(abs(f - c))  # Manhattan distance matrix
        p = f * c  # inner product matrix
        a = 2 * torch.sigmoid(m) * torch.tanh(p)  # dual affinity matrix
        e = a * c + f
        return e

    def forward(self, x):
        # color feature
        x1_2 = self.inc_2(x)
        x2_2 = self.down1_2(x1_2)
        x3_2 = self.down2_2(x2_2)
        x4_2 = self.down3_2(x3_2)
        x5_2 = self.down4_2(x4_2)

        # content feature
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # stage 1
        # color
        x6_2 = self.up1_2(x5_2, x4_2)
        # content
        x6 = self.up1(x5, x4)
        x6_freq = torch.fft.rfft2(self.fpre1(x6), norm='backward')
        _, pha6 = self.frequency_process1(x6_freq, mode='none')
        # color guided structure
        x6_3, x6 = self.CGS1(x6_2, x6)

        # stage 2
        # color
        x7_2 = self.up2_2(x6_3, x3_2)
        # content
        x7 = self.up2(x6, x3)
        x7_freq = torch.fft.rfft2(self.fpre2(x7), norm='backward')
        _, pha7 = self.frequency_process2(x7_freq, mode='none')
        # color guided structure
        x7_3, x7 = self.CGS2(x7_2, x7)

        # stage 3
        # color
        x8_2 = self.up3_2(x7_3, x2_2)
        # content
        x8 = self.up3(x7, x2)
        x8_freq = torch.fft.rfft2(self.fpre3(x8), norm='backward')
        _, pha8 = self.frequency_process3(x8_freq, mode='none')
        # color guided structure
        x8_3, x8 = self.CGS3(x8_2, x8)

        # stage 4
        # color
        x9_2 = self.up4_2(x8_3, x1_2)
        # content
        x9 = self.up4(x8, x1)
        x9_freq = torch.fft.rfft2(self.fpre4(x9), norm='backward')
        _, pha9 = self.frequency_process4(x9_freq, mode='none')
        # color guided structure
        x9_3, x9 = self.CGS4(x9_2, x9)

        x10 = self.pce(x9_3, x9)
        x10 = self.se(x10)
        x10 = self.outc(x10)

        return pha6, pha7, pha8, pha9, x6_2, x7_2, x8_2, x9_2, x10