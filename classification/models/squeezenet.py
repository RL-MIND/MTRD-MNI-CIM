import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import math


class fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(fire, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(expand_planes)
        self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(expand_planes)
        self.relu2 = nn.ReLU(inplace=True)

        # using MSR initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2./n))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        out1 = self.bn2(out1)
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out = torch.cat([self.relu2(out1), self.relu2(out2)], 1)
        #out = self.relu2(out)
        return out


class SqueezeNet(nn.Module):
    def __init__(self,num_classes=10):
        super(SqueezeNet, self).__init__()
        #self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1) # 32
        self.bn1 = nn.BatchNorm2d(96)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 16
        self.fire2 = fire(96, 16, 64)
        self.fire3 = fire(128, 16, 64)
        self.fire4 = fire(128, 32, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 8
        self.fire5 = fire(256, 32, 128)
        self.fire6 = fire(256, 48, 192)
        self.fire7 = fire(384, 48, 192)
        self.fire8 = fire(384, 64, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 4
        self.fire9 = fire(512, 64, 256)
        self.conv2 = nn.Conv2d(512, num_classes, kernel_size=1, stride=1)
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()


    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool2(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.maxpool3(x)
        x = self.fire9(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.GAP(x)
        x = torch.flatten(x, 1)
        x = self.dequant(x)
        return x











class fire_nonid(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(fire_nonid, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(expand_planes)
        self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(expand_planes)
        self.relu2 = nn.ReLU(inplace=True)

        # using MSR initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2./n))

    def forward(self, x, lamda=0.):
        noise1 = torch.normal(0, lamda, size=self.conv1.weight.size()).cuda()
        x = F.conv2d(input=x, weight=torch.exp(noise1) * self.conv1.weight, bias=self.conv1.bias, stride=1)
        #x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        noise2 = torch.normal(0, lamda, size=self.conv2.weight.size()).cuda()
        out1 = F.conv2d(input=x, weight=torch.exp(noise2) * self.conv2.weight, bias=self.conv2.bias, stride=1)
        #out1 = self.conv2(x)
        out1 = self.bn2(out1)
        noise3 = torch.normal(0, lamda, size=self.conv3.weight.size()).cuda()
        out2 = F.conv2d(input=x, weight=torch.exp(noise3) * self.conv3.weight, bias=self.conv3.bias, stride=1, padding=1)
        #out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out = torch.cat([self.relu2(out1), self.relu2(out2)], 1)
        #out = self.relu2(out)
        return out


class SqueezeNet_nonid(nn.Module):
    def __init__(self,num_classes=10):
        super(SqueezeNet_nonid, self).__init__()
        #self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1) # 32
        self.bn1 = nn.BatchNorm2d(96)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 16
        self.fire2 = fire_nonid(96, 16, 64)
        self.fire3 = fire_nonid(128, 16, 64)
        self.fire4 = fire_nonid(128, 32, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 8
        self.fire5 = fire_nonid(256, 32, 128)
        self.fire6 = fire_nonid(256, 48, 192)
        self.fire7 = fire_nonid(384, 48, 192)
        self.fire8 = fire_nonid(384, 64, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 4
        self.fire9 = fire_nonid(512, 64, 256)
        self.conv2 = nn.Conv2d(512, num_classes, kernel_size=1, stride=1)
        self.GAP = nn.AdaptiveAvgPool2d((1,1))


    def forward(self, x, lamda=0.):
        noise1 = torch.normal(0, lamda, size=self.conv1.weight.size()).cuda()
        x = F.conv2d(input=x, weight=torch.exp(noise1) * self.conv1.weight, bias=self.conv1.bias, stride=1, padding=1)
        #x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.fire2(x, lamda)
        x = self.fire3(x, lamda)
        x = self.fire4(x, lamda)
        x = self.maxpool2(x)
        x = self.fire5(x, lamda)
        x = self.fire6(x, lamda)
        x = self.fire7(x, lamda)
        x = self.fire8(x, lamda)
        x = self.maxpool3(x)
        x = self.fire9(x, lamda)
        noise2 = torch.normal(0, lamda, size=self.conv2.weight.size()).cuda()
        x = F.conv2d(input=x, weight=torch.exp(noise2) * self.conv2.weight, bias=self.conv2.bias, stride=1)
        #x = self.conv2(x)
        x = self.relu(x)
        x = self.GAP(x)
        x = torch.flatten(x, 1)
        return x
