import torch.nn as nn
import torch
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self,num_classes=10):
        super(AlexNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.Linear1 = nn.Linear(256, 128)
        self.Linear2 = nn.Linear(128, 32)
        self.Linear3 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.Linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.Linear2(x)
        x = self.relu(x)
        x = self.Linear3(x)
        x = self.dequant(x)

        return x




class AlexNet_nonid(nn.Module):
    def __init__(self,num_classes=10):
        super(AlexNet_nonid, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.Linear1 = nn.Linear(256, 128)
        self.Linear2 = nn.Linear(128, 32)
        self.Linear3 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout()

    def forward(self, x, lamda=0.):
        noise1 = torch.normal(0, lamda, size=self.conv1.weight.size()).cuda()
        x = F.conv2d(input=x, weight=torch.exp(noise1) * self.conv1.weight, bias=self.conv1.bias, stride=4, padding=5)
        #x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        noise2 = torch.normal(0, lamda, size=self.conv2.weight.size()).cuda()
        x = F.conv2d(input=x, weight=torch.exp(noise2) * self.conv2.weight, bias=self.conv2.bias, stride=1, padding=2)
        #x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        noise3 = torch.normal(0, lamda, size=self.conv3.weight.size()).cuda()
        x = F.conv2d(input=x, weight=torch.exp(noise3) * self.conv3.weight, bias=self.conv3.bias, stride=1, padding=1)
        #x = self.conv3(x)
        x = self.relu(x)

        noise4 = torch.normal(0, lamda, size=self.conv4.weight.size()).cuda()
        x = F.conv2d(input=x, weight=torch.exp(noise4) * self.conv4.weight, bias=self.conv4.bias, stride=1, padding=1)
        #x = self.conv4(x)
        x = self.relu(x)

        noise5 = torch.normal(0, lamda, size=self.conv5.weight.size()).cuda()
        x = F.conv2d(input=x, weight=torch.exp(noise5) * self.conv5.weight, bias=self.conv5.bias, stride=1, padding=1)
        #x = self.conv5(x)
        x = self.relu(x)
        x = self.pool(x)

        x = nn.AdaptiveAvgPool2d((1,1))(x)

        x = torch.flatten(x, 1)
        l_noise1 = torch.normal(0, lamda, size=self.Linear1.weight.size()).cuda()
        x = F.linear(input=x, weight=torch.exp(l_noise1) * self.Linear1.weight, bias=self.Linear1.bias)
        #x = self.Linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        l_noise2 = torch.normal(0, lamda, size=self.Linear2.weight.size()).cuda()
        x = F.linear(input=x, weight=torch.exp(l_noise2) * self.Linear2.weight, bias=self.Linear2.bias)
        #x = self.Linear2(x)
        x = self.relu(x)
        l_noise3 = torch.normal(0, lamda, size=self.Linear3.weight.size()).cuda()
        x = F.linear(input=x, weight=torch.exp(l_noise3) * self.Linear3.weight, bias=self.Linear3.bias)
        #x = self.Linear3(x)

        return x

