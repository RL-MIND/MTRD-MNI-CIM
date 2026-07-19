import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        return x
    
class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        #up part of UNET
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))
        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x, lamda=0.0):
        x = self.quant(x)
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        x = self.final_conv(x)
        x = self.dequant(x)
        return x
        

class DoubleConv_nonid(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv_nonid, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)


    def forward(self, x, lamda = 0.5):
        noise1 = torch.normal(0, lamda, size=self.conv1.weight.size()).cuda()
        x = F.conv2d(input=x, weight=torch.exp(noise1) * self.conv1.weight, stride=1, padding=1)
        #x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        noise2 = torch.normal(0, lamda, size=self.conv2.weight.size()).cuda()
        x = F.conv2d(input=x, weight=torch.exp(noise2) * self.conv2.weight, stride=1, padding=1)
        #x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
    
class UNET_nonid(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],):
        super(UNET_nonid, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv_nonid(in_channels, feature))
            in_channels = feature

        #up part of UNET
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv_nonid(feature*2, feature))
        
        self.bottleneck = DoubleConv_nonid(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        #self.quant = torch.ao.quantization.QuantStub()
        #self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x, lamda=0.5):
        #x = self.quant(x)
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        noise3 = torch.normal(0, lamda, size=self.final_conv.weight.size()).cuda()
        x = F.conv2d(input=x, weight=torch.exp(noise3) * self.final_conv.weight, stride=1)
        #x = self.final_conv(x)
        #x = self.dequant(x)
        return x
class DoubleConv_nonid_pcm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv_nonid_pcm, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)


    def forward(self, x, lamda = 0.0):
        weight_copy =self.conv1.weight.clone() 
        max_value = torch.max(weight_copy)
        std = torch.tensor(lamda * max_value)
        noise1 = torch.normal(0, std, size=weight_copy.size()).cuda()    
        x = F.conv2d(input=x, weight=noise1 + self.conv1.weight, stride=1, padding=1)
        #x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        weight_copy =self.conv2.weight.clone() 
        max_value = torch.max(weight_copy)
        std = torch.tensor(lamda * max_value)
        noise2 = torch.normal(0, std, size=weight_copy.size()).cuda()    
        x = F.conv2d(input=x, weight=noise2 + self.conv2.weight, stride=1, padding=1)
        #x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
    
class UNET_nonid_pcm(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],):
        super(UNET_nonid_pcm, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv_nonid_pcm(in_channels, feature))
            in_channels = feature

        #up part of UNET
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv_nonid_pcm(feature*2, feature))
        
        self.bottleneck = DoubleConv_nonid_pcm(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        #self.quant = torch.ao.quantization.QuantStub()
        #self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x, lamda=0.0):
        #x = self.quant(x)
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        weight_copy =self.final_conv.weight.clone() 
        max_value = torch.max(weight_copy)
        std = torch.tensor(lamda * max_value)
        noise3 = torch.normal(0, std, size=weight_copy.size()).cuda()    
        x = F.conv2d(input=x, weight=noise3 + self.final_conv.weight, stride=1)
        #x = self.final_conv(x)
        #x = self.dequant(x)
        return x

class DoubleConv_nonid_src(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv_nonid_src, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)


    def forward(self, x, lamda = 0.0):
        # noise1 = torch.normal(0, lamda, size=self.conv1.weight.size()).cuda()
        # x = F.conv2d(input=x, weight=torch.exp(noise1) * self.conv1.weight, stride=1, padding=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # noise2 = torch.normal(0, lamda, size=self.conv2.weight.size()).cuda()
        # x = F.conv2d(input=x, weight=torch.exp(noise2) * self.conv2.weight, stride=1, padding=1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
    
class UNET_nonid_src(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],):
        super(UNET_nonid_src, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv_nonid_src(in_channels, feature))
            in_channels = feature

        #up part of UNET
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv_nonid_src(feature*2, feature))
        
        self.bottleneck = DoubleConv_nonid_src(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        #self.quant = torch.ao.quantization.QuantStub()
        #self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x, lamda=0.0):
        #x = self.quant(x)
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        #noise3 = torch.normal(0, lamda, size=self.final_conv.weight.size()).cuda()
        #x = F.conv2d(input=x, weight=torch.exp(noise3) * self.final_conv.weight, stride=1)
        x = self.final_conv(x)
        #x = self.dequant(x)
        return x