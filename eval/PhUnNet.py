import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x))
        out = F.leaky_relu(self.conv2(out))
        out += self.shortcut(x)
        return out

class PhUnNet(nn.Module):
    def __init__(self,input_nbr=1):
        super(PhUnNet, self).__init__()

        self.conv1 = nn.Conv2d(input_nbr, 4, kernel_size=3, stride=1, padding=1)
        self.basic11 = BasicBlock(4, 4)
        self.basic12 = BasicBlock(4, 4)
        self.basic13 = BasicBlock(4, 4)
        self.basic14 = BasicBlock(4, 4)
        self.conv2 = nn.Conv2d(4, 1, kernel_size=1, stride=1, padding=0)

        self.conv3 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.basic21 = BasicBlock(8, 8)
        self.basic22 = BasicBlock(8, 8)
        self.basic23 = BasicBlock(8, 8)
        self.basic24 = BasicBlock(8, 8)

        self.conv4 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.basic31 = BasicBlock(16, 16)
        self.basic32 = BasicBlock(16, 16)
        self.basic33 = BasicBlock(16, 16)
        self.basic34 = BasicBlock(16, 16)

        self.conv5 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.basic41 = BasicBlock(32, 32)
        self.basic42 = BasicBlock(32, 32)
        self.basic43 = BasicBlock(32, 32)
        self.basic44 = BasicBlock(32, 32)

        self.conv6 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.uppool1 = nn.ConvTranspose2d(64, 16, 2, stride=2, bias=True)

        self.conv7 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.uppool2 = nn.ConvTranspose2d(32, 8, 2, stride=2, bias=True)

        self.conv8 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.uppool3 = nn.ConvTranspose2d(16, 4, 2, stride=2, bias=True)

        self.conv9 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)


    def forward(self, x):

        out = self.conv1(x)
        out = self.basic11(out)
        out = self.basic12(out)
        out = self.basic13(out)
        out = self.basic14(out)
        outres = self.conv2(out)

        out = self.conv3(out)
        out = self.pool1(out)
        out = self.basic21(out)
        out = self.basic22(out)
        out = self.basic23(out)
        out = self.basic24(out)

        out = self.conv4(out)
        out = self.pool2(out)
        out = self.basic31(out)
        out = self.basic32(out)
        out = self.basic33(out)
        out = self.basic34(out)

        out = self.conv5(out)
        out = self.pool3(out)
        out = self.basic41(out)
        out = self.basic42(out)
        out = self.basic43(out)
        out = self.basic44(out)

        out = F.leaky_relu(self.conv6(out))
        out = self.uppool1(out)

        out = F.leaky_relu(self.conv7(out))
        out = self.uppool2(out)

        out = F.leaky_relu(self.conv8(out))
        out = self.uppool3(out)

        out = self.conv9(out)
        out += outres

        return out