import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock1(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock1, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2((self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock2(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock2, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2((self.conv2(out))))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def down_creator(in_channels, out_channels):
    return nn.Sequential(
        nn.MaxPool2d(kernel_size=2, stride=2),
        BasicBlock2(in_channels, out_channels)
    )


def conv_up(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block1 = BasicBlock1(3, 64)
        self.block2 = BasicBlock1(64, 128)

        self.down1 = down_creator(128, 256)
        self.down2 = down_creator(256, 512)
        self.down3 = down_creator(512, 512)
        self.down4 = down_creator(512, 512)

        self.up1 = conv_up(512, 512)
        self.upblock1 = BasicBlock2(1024, 512)

        self.up2 = conv_up(512, 512)
        self.upblock2 = BasicBlock2(1024, 512)

        self.up3 = conv_up(512, 256)
        self.upblock3 = BasicBlock2(512, 256)

        self.up4 = conv_up(256, 128)
        self.upblock4 = BasicBlock2(256, 128)

        self.up5 = conv_up(128, 64)

        self.blockfinal1 = BasicBlock1(128, 64)
        self.finalconv = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, image):
        # encoder
        x1 = self.block1(image)

        x2 = self.max_pool_2x2(x1)
        x3 = self.block2(x2)

        x4 = self.down1(x3)
        x5 = self.down2(x4)
        x6 = self.down3(x5)
        x7 = self.down4(x6)

        # decoder
        y1 = self.up1(x7)
        y1 = torch.cat([y1, x6], 1)
        y1 = self.upblock1(y1)

        y2 = self.up2(y1)
        y2 = torch.cat([y2, x5], 1)
        y2 = self.upblock2(y2)

        y3 = self.up3(y2)
        y3 = torch.cat([y3, x4], 1)
        y3 = self.upblock3(y3)

        y4 = self.up4(y3)
        y4 = torch.cat([y4, x3], 1)
        y4 = self.upblock4(y4)

        y5 = self.up5(y4)
        y5 = torch.cat([y5, x1], 1)

        y5 = self.blockfinal1(y5)
        out = self.finalconv(y5)

        return out