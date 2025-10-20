import torch
import torch.nn as nn
import torch.nn.functional as F



class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(features)
        self.conv3 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn3 = nn.BatchNorm2d(features)
        self.conv4 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn4 = nn.BatchNorm2d(features)
        self.conv5 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn5 = nn.BatchNorm2d(features)
        self.conv6 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn6 = nn.BatchNorm2d(features)
        self.conv7 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn7 = nn.BatchNorm2d(features)
        self.conv8 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn8 = nn.BatchNorm2d(features)
        self.conv9 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn9 = nn.BatchNorm2d(features)
        self.conv10 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn10 = nn.BatchNorm2d(features)
        self.conv11 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn11 = nn.BatchNorm2d(features)
        self.conv12 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn12 = nn.BatchNorm2d(features)
        self.conv13 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn13 = nn.BatchNorm2d(features)
        self.conv14 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn14 = nn.BatchNorm2d(features)
        self.conv15 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn15 = nn.BatchNorm2d(features)
        self.conv16 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn16 = nn.BatchNorm2d(features)
        self.conv17 = nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x, lamda=0.):
        x = self.quant(x)
        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)

        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu(out)

        out = self.conv6(out)
        out = self.bn6(out)
        out = self.relu(out)

        out = self.conv7(out)
        out = self.bn7(out)
        out = self.relu(out)

        out = self.conv8(out)
        out = self.bn8(out)
        out = self.relu(out)

        out = self.conv9(out)
        out = self.bn9(out)
        out = self.relu(out)

        out = self.conv10(out)
        out = self.bn10(out)
        out = self.relu(out)

        out = self.conv11(out)
        out = self.bn11(out)
        out = self.relu(out)

        out = self.conv12(out)
        out = self.bn12(out)
        out = self.relu(out)

        out = self.conv13(out)
        out = self.bn13(out)
        out = self.relu(out)

        out = self.conv14(out)
        out = self.bn14(out)
        out = self.relu(out)

        out = self.conv15(out)
        out = self.bn15(out)
        out = self.relu(out)

        out = self.conv16(out)
        out = self.bn16(out)
        out = self.relu(out)

        out = self.conv17(out)

        out = self.dequant(out)
        
        
        return out





class DnCNN_bak(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN_bak, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
    def forward(self, x):
        x = self.quant(x)
        out = self.dncnn(x)
        out = self.dequant(out)
        return out

class DnCNN_nonid(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN_nonid, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(features)
        self.conv3 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn3 = nn.BatchNorm2d(features)
        self.conv4 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn4 = nn.BatchNorm2d(features)
        self.conv5 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn5 = nn.BatchNorm2d(features)
        self.conv6 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn6 = nn.BatchNorm2d(features)
        self.conv7 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn7 = nn.BatchNorm2d(features)
        self.conv8 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn8 = nn.BatchNorm2d(features)
        self.conv9 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn9 = nn.BatchNorm2d(features)
        self.conv10 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn10 = nn.BatchNorm2d(features)
        self.conv11 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn11 = nn.BatchNorm2d(features)
        self.conv12 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn12 = nn.BatchNorm2d(features)
        self.conv13 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn13 = nn.BatchNorm2d(features)
        self.conv14 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn14 = nn.BatchNorm2d(features)
        self.conv15 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn15 = nn.BatchNorm2d(features)
        self.conv16 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn16 = nn.BatchNorm2d(features)
        self.conv17 = nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, lamda=0.):
        noise1 = torch.normal(0, lamda, size=self.conv1.weight.size()).cuda()
        out = F.conv2d(input=x, weight=torch.exp(noise1) * self.conv1.weight, stride=1, padding=1)
        #out = self.conv1(x)
        out = self.relu(out)

        noise2 = torch.normal(0, lamda, size=self.conv2.weight.size()).cuda()
        out = F.conv2d(input=out, weight=torch.exp(noise2) * self.conv2.weight, stride=1, padding=1)
        #out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        noise3 = torch.normal(0, lamda, size=self.conv3.weight.size()).cuda()
        out = F.conv2d(input=out, weight=torch.exp(noise3) * self.conv3.weight, stride=1, padding=1)
        #out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        noise4 = torch.normal(0, lamda, size=self.conv4.weight.size()).cuda()
        out = F.conv2d(input=out, weight=torch.exp(noise4) * self.conv4.weight, stride=1, padding=1)
        #out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)

        noise5 = torch.normal(0, lamda, size=self.conv5.weight.size()).cuda()
        out = F.conv2d(input=out, weight=torch.exp(noise5) * self.conv5.weight, stride=1, padding=1)
        #out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu(out)

        noise6 = torch.normal(0, lamda, size=self.conv6.weight.size()).cuda()
        out = F.conv2d(input=out, weight=torch.exp(noise6) * self.conv6.weight, stride=1, padding=1)
        #out = self.conv6(out)
        out = self.bn6(out)
        out = self.relu(out)

        noise7 = torch.normal(0, lamda, size=self.conv7.weight.size()).cuda()
        out = F.conv2d(input=out, weight=torch.exp(noise7) * self.conv7.weight, stride=1, padding=1)
        #out = self.conv7(out)
        out = self.bn7(out)
        out = self.relu(out)

        noise8 = torch.normal(0, lamda, size=self.conv8.weight.size()).cuda()
        out = F.conv2d(input=out, weight=torch.exp(noise8) * self.conv8.weight, stride=1, padding=1)
        #out = self.conv8(out)
        out = self.bn8(out)
        out = self.relu(out)

        noise9 = torch.normal(0, lamda, size=self.conv9.weight.size()).cuda()
        out = F.conv2d(input=out, weight=torch.exp(noise9) * self.conv9.weight, stride=1, padding=1)
        #out = self.conv9(out)
        out = self.bn9(out)
        out = self.relu(out)

        noise10 = torch.normal(0, lamda, size=self.conv10.weight.size()).cuda()
        out = F.conv2d(input=out, weight=torch.exp(noise10) * self.conv10.weight, stride=1, padding=1)
        #out = self.conv10(out)
        out = self.bn10(out)
        out = self.relu(out)

        noise11 = torch.normal(0, lamda, size=self.conv11.weight.size()).cuda()
        out = F.conv2d(input=out, weight=torch.exp(noise11) * self.conv11.weight, stride=1, padding=1)
        #out = self.conv11(out)
        out = self.bn11(out)
        out = self.relu(out)

        noise12 = torch.normal(0, lamda, size=self.conv12.weight.size()).cuda()
        out = F.conv2d(input=out, weight=torch.exp(noise12) * self.conv12.weight, stride=1, padding=1)
        #out = self.conv12(out)
        out = self.bn12(out)
        out = self.relu(out)

        noise13 = torch.normal(0, lamda, size=self.conv13.weight.size()).cuda()
        out = F.conv2d(input=out, weight=torch.exp(noise13) * self.conv13.weight, stride=1, padding=1)
        #out = self.conv13(out)
        out = self.bn13(out)
        out = self.relu(out)

        noise14 = torch.normal(0, lamda, size=self.conv14.weight.size()).cuda()
        out = F.conv2d(input=out, weight=torch.exp(noise14) * self.conv14.weight, stride=1, padding=1)
        #out = self.conv14(out)
        out = self.bn14(out)
        out = self.relu(out)

        noise15 = torch.normal(0, lamda, size=self.conv15.weight.size()).cuda()
        out = F.conv2d(input=out, weight=torch.exp(noise15) * self.conv15.weight, stride=1, padding=1)
        #out = self.conv15(out)
        out = self.bn15(out)
        out = self.relu(out)

        noise16 = torch.normal(0, lamda, size=self.conv16.weight.size()).cuda()
        out = F.conv2d(input=out, weight=torch.exp(noise16) * self.conv16.weight, stride=1, padding=1)
        #out = self.conv16(out)
        out = self.bn16(out)
        out = self.relu(out)

        noise17 = torch.normal(0, lamda, size=self.conv17.weight.size()).cuda()
        out = F.conv2d(input=out, weight=torch.exp(noise17) * self.conv17.weight, stride=1, padding=1)
        #out = self.conv17(out)
        
        
        return out


class DnCNN_nonid_pcm(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN_nonid_pcm, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(features)
        self.conv3 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn3 = nn.BatchNorm2d(features)
        self.conv4 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn4 = nn.BatchNorm2d(features)
        self.conv5 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn5 = nn.BatchNorm2d(features)
        self.conv6 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn6 = nn.BatchNorm2d(features)
        self.conv7 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn7 = nn.BatchNorm2d(features)
        self.conv8 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn8 = nn.BatchNorm2d(features)
        self.conv9 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn9 = nn.BatchNorm2d(features)
        self.conv10 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn10 = nn.BatchNorm2d(features)
        self.conv11 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn11 = nn.BatchNorm2d(features)
        self.conv12 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn12 = nn.BatchNorm2d(features)
        self.conv13 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn13 = nn.BatchNorm2d(features)
        self.conv14 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn14 = nn.BatchNorm2d(features)
        self.conv15 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn15 = nn.BatchNorm2d(features)
        self.conv16 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn16 = nn.BatchNorm2d(features)
        self.conv17 = nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, lamda=0.):
        weight_copy =self.conv1.weight.clone() 
        max_value = torch.max(weight_copy)
        std = torch.tensor(lamda * max_value)
        noise1 = torch.normal(0, std, size=weight_copy.size()).cuda()  
        out = F.conv2d(input=x, weight= noise1 + self.conv1.weight, stride=1, padding=1)
        #out = self.conv1(x)
        out = self.relu(out)

        weight_copy =self.conv2.weight.clone() 
        max_value = torch.max(weight_copy)
        std = torch.tensor(lamda * max_value)
        noise2 = torch.normal(0, std, size=weight_copy.size()).cuda() 
        out = F.conv2d(input=out, weight= noise2 + self.conv2.weight, stride=1, padding=1)
        #out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        weight_copy =self.conv3.weight.clone() 
        max_value = torch.max(weight_copy)
        std = torch.tensor(lamda * max_value)
        noise3 = torch.normal(0, std, size=weight_copy.size()).cuda() 
        out = F.conv2d(input=out, weight= noise3 + self.conv3.weight, stride=1, padding=1)
        #out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        weight_copy =self.conv4.weight.clone() 
        max_value = torch.max(weight_copy)
        std = torch.tensor(lamda * max_value)
        noise4 = torch.normal(0, std, size=weight_copy.size()).cuda() 
        out = F.conv2d(input=out, weight= noise4 + self.conv4.weight, stride=1, padding=1)
        #out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)

        weight_copy =self.conv5.weight.clone() 
        max_value = torch.max(weight_copy)
        std = torch.tensor(lamda * max_value)
        noise5 = torch.normal(0, std, size=weight_copy.size()).cuda() 
        out = F.conv2d(input=out, weight= noise5 + self.conv5.weight, stride=1, padding=1)
        #out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu(out)

        weight_copy =self.conv6.weight.clone() 
        max_value = torch.max(weight_copy)
        std = torch.tensor(lamda * max_value)
        noise6 = torch.normal(0, std, size=weight_copy.size()).cuda() 
        out = F.conv2d(input=out, weight= noise6 + self.conv6.weight, stride=1, padding=1)
        #out = self.conv6(out)
        out = self.bn6(out)
        out = self.relu(out)

        weight_copy =self.conv7.weight.clone() 
        max_value = torch.max(weight_copy)
        std = torch.tensor(lamda * max_value)
        noise7 = torch.normal(0, std, size=weight_copy.size()).cuda() 
        out = F.conv2d(input=out, weight= noise7 + self.conv7.weight, stride=1, padding=1)
        #out = self.conv7(out)
        out = self.bn7(out)
        out = self.relu(out)

        weight_copy =self.conv8.weight.clone() 
        max_value = torch.max(weight_copy)
        std = torch.tensor(lamda * max_value)
        noise8 = torch.normal(0, std, size=weight_copy.size()).cuda() 
        out = F.conv2d(input=out, weight=noise8 + self.conv8.weight, stride=1, padding=1)
        #out = self.conv8(out)
        out = self.bn8(out)
        out = self.relu(out)

        weight_copy =self.conv9.weight.clone() 
        max_value = torch.max(weight_copy)
        std = torch.tensor(lamda * max_value)
        noise9 = torch.normal(0, std, size=weight_copy.size()).cuda() 
        out = F.conv2d(input=out, weight= noise9 + self.conv9.weight, stride=1, padding=1)
        #out = self.conv9(out)
        out = self.bn9(out)
        out = self.relu(out)

        weight_copy =self.conv10.weight.clone() 
        max_value = torch.max(weight_copy)
        std = torch.tensor(lamda * max_value)
        noise10 = torch.normal(0, std, size=weight_copy.size()).cuda() 
        out = F.conv2d(input=out, weight= noise10 + self.conv10.weight, stride=1, padding=1)
        #out = self.conv10(out)
        out = self.bn10(out)
        out = self.relu(out)

        weight_copy =self.conv11.weight.clone() 
        max_value = torch.max(weight_copy)
        std = torch.tensor(lamda * max_value)
        noise11 = torch.normal(0, std, size=weight_copy.size()).cuda() 
        out = F.conv2d(input=out, weight= noise11 + self.conv11.weight, stride=1, padding=1)
        #out = self.conv11(out)
        out = self.bn11(out)
        out = self.relu(out)

        weight_copy =self.conv12.weight.clone() 
        max_value = torch.max(weight_copy)
        std = torch.tensor(lamda * max_value)
        noise12 = torch.normal(0, std, size=weight_copy.size()).cuda() 
        out = F.conv2d(input=out, weight= noise12 + self.conv12.weight, stride=1, padding=1)
        #out = self.conv12(out)
        out = self.bn12(out)
        out = self.relu(out)

        weight_copy =self.conv13.weight.clone() 
        max_value = torch.max(weight_copy)
        std = torch.tensor(lamda * max_value)
        noise13 = torch.normal(0, std, size=weight_copy.size()).cuda() 
        out = F.conv2d(input=out, weight= noise13 + self.conv13.weight, stride=1, padding=1)
        #out = self.conv13(out)
        out = self.bn13(out)
        out = self.relu(out)

        weight_copy =self.conv14.weight.clone() 
        max_value = torch.max(weight_copy)
        std = torch.tensor(lamda * max_value)
        noise14 = torch.normal(0, std, size=weight_copy.size()).cuda() 
        out = F.conv2d(input=out, weight= noise14 + self.conv14.weight, stride=1, padding=1)
        #out = self.conv14(out)
        out = self.bn14(out)
        out = self.relu(out)

        weight_copy =self.conv15.weight.clone() 
        max_value = torch.max(weight_copy)
        std = torch.tensor(lamda * max_value)
        noise15 = torch.normal(0, std, size=weight_copy.size()).cuda() 
        out = F.conv2d(input=out, weight= noise15 + self.conv15.weight, stride=1, padding=1)
        #out = self.conv15(out)
        out = self.bn15(out)
        out = self.relu(out)

        weight_copy =self.conv16.weight.clone() 
        max_value = torch.max(weight_copy)
        std = torch.tensor(lamda * max_value)
        noise16 = torch.normal(0, std, size=weight_copy.size()).cuda() 
        out = F.conv2d(input=out, weight= noise16 + self.conv16.weight, stride=1, padding=1)
        #out = self.conv16(out)
        out = self.bn16(out)
        out = self.relu(out)

        weight_copy =self.conv17.weight.clone() 
        max_value = torch.max(weight_copy)
        std = torch.tensor(lamda * max_value)
        noise17 = torch.normal(0, std, size=weight_copy.size()).cuda() 
        out = F.conv2d(input=out, weight=noise17 + self.conv17.weight, stride=1, padding=1)
        #out = self.conv17(out)
        
        
        return out



class DnCNN_smi(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN_smi, self).__init__()
        kernel_size = 3
        padding = 1
        features = 16
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(features)
        self.conv3 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn3 = nn.BatchNorm2d(features)
        self.conv4 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn4 = nn.BatchNorm2d(features)
        self.conv5 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn5 = nn.BatchNorm2d(features)
        self.conv6 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn6 = nn.BatchNorm2d(features)
        self.conv7 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn7 = nn.BatchNorm2d(features)
        self.conv8 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn8 = nn.BatchNorm2d(features)
        self.conv9 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn9 = nn.BatchNorm2d(features)
        self.conv10 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn10 = nn.BatchNorm2d(features)
        self.conv11 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn11 = nn.BatchNorm2d(features)
        self.conv12 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn12 = nn.BatchNorm2d(features)
        self.conv13 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn13 = nn.BatchNorm2d(features)
        self.conv14 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn14 = nn.BatchNorm2d(features)
        self.conv15 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn15 = nn.BatchNorm2d(features)
        self.conv16 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn16 = nn.BatchNorm2d(features)
        self.conv17 = nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x, lamda=0.):
        x = self.quant(x)
        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)

        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu(out)

        out = self.conv6(out)
        out = self.bn6(out)
        out = self.relu(out)

        out = self.conv7(out)
        out = self.bn7(out)
        out = self.relu(out)

        out = self.conv8(out)
        out = self.bn8(out)
        out = self.relu(out)

        out = self.conv9(out)
        out = self.bn9(out)
        out = self.relu(out)

        out = self.conv10(out)
        out = self.bn10(out)
        out = self.relu(out)

        out = self.conv11(out)
        out = self.bn11(out)
        out = self.relu(out)

        out = self.conv12(out)
        out = self.bn12(out)
        out = self.relu(out)

        out = self.conv13(out)
        out = self.bn13(out)
        out = self.relu(out)

        out = self.conv14(out)
        out = self.bn14(out)
        out = self.relu(out)

        out = self.conv15(out)
        out = self.bn15(out)
        out = self.relu(out)

        out = self.conv16(out)
        out = self.bn16(out)
        out = self.relu(out)

        out = self.conv17(out)

        out = self.dequant(out)
        
        
        return out