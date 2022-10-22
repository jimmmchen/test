import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1=conv3x3(in_channels, out_channels, stride)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=conv3x3(out_channels, out_channels)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.downsample=downsample

    def forward(self, x):
        residual=x
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        if self.downsample:
            residual=self.downsample(x)
        out+=residual
        out=self.relu(out)
        return out

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1=nn.Conv2d(in_channels, out_channels)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.conv2=conv3x3(out_channels, out_channels, stride)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.conv2=nn.Conv2d(out_channels, out_channels)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample

    def forward(self, x):
        residual=x
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.conv2(out)
        out=self.bn2(out)
        out=self.conv3(out)
        out=self.bn3(out)
        out=self.relu(out)
        if self.downsample:
            residual=self.downsample(x)
        out+=residual
        out=self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super(ResNet, self).__init__()
        self.in_channels=64
        self.conv=nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpol=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1=nn.BatchNorm2d(64)
        self.relu1=nn.ReLU(inplace=True)
        self.layer1=self.make_layer(block, 64, layers[0])
        self.layer2=self.make_layer(block, 128, layers[1], 2)
        self.layer3=self.make_layer(block, 256, layers[2], 2)
        self.avg_pool=nn.AvgPool2d(7)
        self.fc=nn.Linear(256, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample=None
        if stride!=1 or self.in_channels!=out_channels:
            downsample=nn.Sequential(conv3x3(self.in_channels, out_channels, stride=stride), nn.BatchNorm2d(out_channels))
        layers=[]
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels=out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    def forward(self, x):
        out=self.conv(x)
        out=self.bn(out)
        out=self.relu(out)
        out=self.maxpol(out)
        out=self.bn1(out)
        out=self.relu1(out)
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.avg_pool(out)
        out=out.view(out.size(0), -1)
        out=self.fc(out)
        return out
