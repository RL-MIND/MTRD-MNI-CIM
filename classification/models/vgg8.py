import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class vgg8(nn.Module):
    def __init__(self,num_classes=10):
        super(vgg8, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(512)

        self.lonear1 = nn.Linear(512, 256)
        self.lbn1 = nn.BatchNorm1d(256)
        self.lonear2 = nn.Linear(256, 128)
        self.lbn2 = nn.BatchNorm1d(128)
        self.lonear3 = nn.Linear(128, num_classes)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool2d(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool2d(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.maxpool2d(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)
        x = self.maxpool2d(x)

        x = self.conv11(x)
        x = self.bn11(x)
        x = self.relu(x)
        x = nn.AvgPool2d(2)(x)

        x = torch.flatten(x, 1)
        
        x = self.lonear1(x)

        x = self.lbn1(x)
        
        x = self.relu(x)
        x = self.lonear2(x)


        x = self.lbn2(x)

        x = self.relu(x)
        x = self.lonear3(x)

        return x

class vgg8_quant(nn.Module):
    def __init__(self,num_classes=10):
        super(vgg8_quant, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(512)

        self.lonear1 = nn.Linear(512, 256)
        self.lbn1 = nn.BatchNorm1d(256)
        self.lonear2 = nn.Linear(256, 128)
        self.lbn2 = nn.BatchNorm1d(128)
        self.lonear3 = nn.Linear(128, num_classes)

        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

        self._initialize_weights()
        

    def forward(self, x):
        x = self.quant(x)
        # noise1 = self.conv1.weight*torch.normal(0., torch.full(self.conv1.weight.size(),lamda)).cuda()
        # x = F.conv2d(input=x, weight=noise1 + self.conv1.weight, bias=self.conv1.bias, stride=1, padding=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool2d(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool2d(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.maxpool2d(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)
        x = self.maxpool2d(x)

        x = self.conv11(x)
        x = self.bn11(x)
        x = self.relu(x)
        x = nn.AvgPool2d(2)(x)

        x = torch.flatten(x, 1)
        
        x = self.lonear1(x)
        x = self.dequant(x)

        x = self.lbn1(x)
        
        x = self.quant(x)
        x = self.relu(x)
        x = self.lonear2(x)
        x = self.dequant(x)


        x = self.lbn2(x)

        x = self.quant(x)
        x = self.relu(x)
        x = self.lonear3(x)
        x = self.dequant(x)

        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class vgg8_nonid(nn.Module):
    def __init__(self,num_classes=10):
        super(vgg8_nonid, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(512)

        self.lonear1 = nn.Linear(512, 256)
        self.lbn1 = nn.BatchNorm1d(256)
        self.lonear2 = nn.Linear(256, 128)
        self.lbn2 = nn.BatchNorm1d(128)
        self.lonear3 = nn.Linear(128, num_classes)

        self._initialize_weights()
        

    def forward(self, x, lamda=0.):
        # noise1 = self.conv1.weight*torch.normal(0., torch.full(self.conv1.weight.size(),lamda)).cuda()
        # x = F.conv2d(input=x, weight=noise1 + self.conv1.weight, bias=self.conv1.bias, stride=1, padding=1)
        noise1 = torch.normal(0, lamda, size=self.conv1.weight.size()).cuda()
        x = F.conv2d(input=x, weight=torch.exp(noise1) * self.conv1.weight, bias=self.conv1.bias, stride=1, padding=1)
        #x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool2d(x)

        noise3 = torch.normal(0, lamda, size=self.conv3.weight.size()).cuda()
        x = F.conv2d(input=x, weight=torch.exp(noise3) * self.conv3.weight, bias=self.conv3.bias, stride=1, padding=1)
        #x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool2d(x)

        noise5 = torch.normal(0, lamda, size=self.conv5.weight.size()).cuda()
        x = F.conv2d(input=x, weight=torch.exp(noise5) * self.conv5.weight, bias=self.conv5.bias, stride=1, padding=1)
        #x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.maxpool2d(x)

        noise8 = torch.normal(0, lamda, size=self.conv8.weight.size()).cuda()
        x = F.conv2d(input=x, weight=torch.exp(noise8) * self.conv8.weight, bias=self.conv8.bias, stride=1, padding=1)
        #x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)
        x = self.maxpool2d(x)

        noise11 = torch.normal(0, lamda, size=self.conv11.weight.size()).cuda()
        x = F.conv2d(input=x, weight=torch.exp(noise11) * self.conv11.weight, bias=self.conv11.bias, stride=1, padding=1)
        #x = self.conv11(x)
        x = self.bn11(x)
        x = self.relu(x)
        #x = nn.AvgPool2d(2)(x)
        x = nn.AdaptiveAvgPool2d((1,1))(x)

        x = torch.flatten(x, 1)
        
        l_noise1 = torch.normal(0, lamda, size=self.lonear1.weight.size()).cuda()
        x = F.linear(input=x, weight=torch.exp(l_noise1) * self.lonear1.weight, bias=self.lonear1.bias)
        #x = self.lonear1(x)
        x = self.lbn1(x)
        x = self.relu(x)
        l_noise2 = torch.normal(0, lamda, size=self.lonear2.weight.size()).cuda()
        x = F.linear(input=x, weight=torch.exp(l_noise2) * self.lonear2.weight, bias=self.lonear2.bias)
        #x = self.lonear2(x)
        x = self.lbn2(x)
        x = self.relu(x)
        l_noise3 = torch.normal(0, lamda, size=self.lonear3.weight.size()).cuda()
        x = F.linear(input=x, weight=torch.exp(l_noise3) * self.lonear3.weight, bias=self.lonear3.bias)
        #x = self.lonear3(x)

        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class vgg8_nonid_pcm(nn.Module):
    def __init__(self,num_classes=10):
        super(vgg8_nonid_pcm, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(512)

        self.lonear1 = nn.Linear(512, 256)
        self.lbn1 = nn.BatchNorm1d(256)
        self.lonear2 = nn.Linear(256, 128)
        self.lbn2 = nn.BatchNorm1d(128)
        self.lonear3 = nn.Linear(128, num_classes)

        self._initialize_weights()
        

    def forward(self, x, lamda=0.):
        weight_copy =self.conv1.weight.clone() 
        max_value = torch.max(weight_copy)
        std = torch.tensor(lamda * max_value)
        noise1 = torch.normal(0, std, size=weight_copy.size()).cuda()    
        x = F.conv2d(input=x, weight=noise1 + self.conv1.weight, bias=self.conv1.bias, stride=1, padding=1)
        #x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool2d(x)

        weight_copy =self.conv3.weight.clone() 
        max_value = torch.max(weight_copy)
        std = torch.tensor(lamda * max_value)
        noise3 = torch.normal(0, std, size=weight_copy.size()).cuda()
        x = F.conv2d(input=x, weight=noise3 + self.conv3.weight, bias=self.conv3.bias, stride=1, padding=1)
        #x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool2d(x)

        weight_copy =self.conv5.weight.clone() 
        max_value = torch.max(weight_copy)
        std = torch.tensor(lamda * max_value)
        noise5 = torch.normal(0, std, size=weight_copy.size()).cuda()
        x = F.conv2d(input=x, weight= noise5 + self.conv5.weight, bias=self.conv5.bias, stride=1, padding=1)
        #x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.maxpool2d(x)

        weight_copy =self.conv8.weight.clone() 
        max_value = torch.max(weight_copy)
        std = torch.tensor(lamda * max_value)
        noise8 = torch.normal(0, std, size=weight_copy.size()).cuda()
        x = F.conv2d(input=x, weight= noise8 + self.conv8.weight, bias=self.conv8.bias, stride=1, padding=1)
        #x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)
        x = self.maxpool2d(x)

        weight_copy =self.conv11.weight.clone() 
        max_value = torch.max(weight_copy)
        std = torch.tensor(lamda * max_value)
        noise11 = torch.normal(0, std, size=weight_copy.size()).cuda()
        x = F.conv2d(input=x, weight= noise11 + self.conv11.weight, bias=self.conv11.bias, stride=1, padding=1)
        #x = self.conv11(x)
        x = self.bn11(x)
        x = self.relu(x)
        #x = nn.AvgPool2d(2)(x)
        x = nn.AdaptiveAvgPool2d((1,1))(x)

        x = torch.flatten(x, 1)
        
        weight_copy =self.lonear1.weight.clone() 
        max_value = torch.max(weight_copy)
        std = torch.tensor(lamda * max_value)
        l_noise1 = torch.normal(0, std, size=weight_copy.size()).cuda()
        x = F.linear(input=x, weight= l_noise1 + self.lonear1.weight, bias=self.lonear1.bias)
        #x = self.lonear1(x)
        x = self.lbn1(x)
        x = self.relu(x)
        weight_copy =self.lonear2.weight.clone() 
        max_value = torch.max(weight_copy)
        std = torch.tensor(lamda * max_value)
        l_noise2 = torch.normal(0, std, size=weight_copy.size()).cuda()
        x = F.linear(input=x, weight= l_noise2 + self.lonear2.weight, bias=self.lonear2.bias)
        #x = self.lonear2(x)
        x = self.lbn2(x)
        x = self.relu(x)
        weight_copy =self.lonear3.weight.clone() 
        max_value = torch.max(weight_copy)
        std = torch.tensor(lamda * max_value)
        l_noise3 = torch.normal(0, std, size=weight_copy.size()).cuda()
        x = F.linear(input=x, weight= l_noise3 + self.lonear3.weight, bias=self.lonear3.bias)
        
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()