'''Train CIFAR10 with PyTorch.'''
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import os
import argparse
import torch.quantization as quant

from models import *
from utils import progress_bar

import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 test')
parser.add_argument('--precheckpoint_root', default='checkpoint/cifar10/vgg8/ckpt_cifar10_nonid_0.pth', help='Load precheckpoint', type=str)
parser.add_argument('--noise', default=0.0, type=float, help='noise')
parser.add_argument('--quant_bit', default=8, type=int, help='quant')
parser.add_argument('--model_name', default='vgg8', help='choice a model to train and eval. eg: alenet, vgg16', type=str)
parser.add_argument('--data_root', default='data/cifar10_dataset', help='Path to the train dataset', type=str)
parser.add_argument('--num_workers', default=0, help='number of workers', type=int)
parser.add_argument('--batch_size', default=64, help='number of batch size', type=int)
parser.add_argument('--dataset', type=str, default='cifar10',help='training dataset (default: cifar100)')
args = parser.parse_args()

device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')
num_c=0

kwargs = {'num_workers': 1, 'pin_memory': True} 
if args.dataset == 'cifar10':
    num_c=10
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
else:
    num_c=100
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)


print('train dataset len: {}'.format(len(train_loader.dataset)))

print('val dataset len: {}'.format(len(test_loader.dataset)))

# Model
print('==> Building model..')
if args.model_name == "vgg16":
    net = vgg16()
elif args.model_name == "vgg8": #ok
    net0 = vgg8(num_classes= num_c)
    net1 = vgg8(num_classes= num_c)
    net2 = vgg8(num_classes= num_c)
    net3 = vgg8(num_classes= num_c)
    net4 = vgg8(num_classes= num_c)
    net5 = vgg8(num_classes= num_c)
elif args.model_name == "squeezenet": #ok
    net = squeezenet()
elif args.model_name == "ResNet18": #ok
    net = ResNet18()
elif args.model_name == "ResNet34":  #ok
    net = ResNet34()
elif args.model_name == "ResNet50": #ok
    net = ResNet50()
elif args.model_name == "PreActResNet18": #ok
    net = PreActResNet18()
elif args.model_name == "PreActResNet34": #ok
    net = PreActResNet34()
elif args.model_name == "PreActResNet50": #ok
    net = PreActResNet50()
elif args.model_name == "PreActResNet101": #ok
    net = PreActResNet101()
elif args.model_name == "PreActResNet152": #ok
    net = PreActResNet152()
elif args.model_name == "GoogLeNet": #ok
    net = GoogLeNet()
elif args.model_name == "DenseNet121": #ok
    net = DenseNet121()
elif args.model_name == "ResNeXt29_2x64d": #ok
    net = ResNeXt29_2x64d()
elif args.model_name == "MobileNet": #ok
    net = MobileNet()
elif args.model_name == "MobileNetV2": #ok
    net = MobileNetV2()
elif args.model_name == "DPN92": #ok
    net = DPN92()
elif args.model_name == "ShuffleNetG2":
    net = ShuffleNetG2()
elif args.model_name == "SENet18":  #ok
    net = SENet18()
elif args.model_name == "ShuffleNetV2":  #ok
    net = ShuffleNetV2(1)
elif args.model_name == "EfficientNetB0":  #ok
    net = EfficientNetB0()
elif args.model_name == "RegNetX_200MF":  #ok
    net = RegNetX_200MF()
elif args.model_name == "SimpleDLA":  #ok
    net = SimpleDLA()
elif args.model_name == "LeNet":  #ok
    net = LeNet()
elif args.model_name == "AlexNet":  #ok
    net = AlexNet()
elif args.model_name=="vit_small":
    net = ViT( image_size = 32, patch_size = 4, num_classes = 10, dim = int(512), depth = 6, heads = 8, mlp_dim = 512, dropout = 0.1, emb_dropout = 0.1)
else:
    print("Error: model name not define! Exit...")
    exit(1)

net0 = net0.to(device)
net1 = net1.to(device)
net2 = net2.to(device)
net3 = net3.to(device)
net4 = net4.to(device)
net5 = net5.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True


# Load checkpoint.
print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load(args.precheckpoint_root)
net0.load_state_dict(checkpoint)
net1.load_state_dict(checkpoint)
net2.load_state_dict(checkpoint)
net3.load_state_dict(checkpoint)
net4.load_state_dict(checkpoint)
net5.load_state_dict(checkpoint)

def add_noise_to_conv_layers(model, std=0.1):
    for module in model.children():
        if isinstance(module, nn.Conv2d):
            # Add Gaussian noise to convolutional layer weights and biases
            with torch.no_grad():
                weight_copy = module.weight.data.clone()
                noise = torch.normal(0, std, size=weight_copy.size())             
                module.weight.data= torch.exp(noise) * weight_copy
                #print(module)
        elif isinstance(module, nn.Linear):
            # Add Gaussian noise to linear layer weights and biases
            # Add Gaussian noise to convolutional layer weights and biases
            with torch.no_grad():
                weight_copy = module.weight.data.clone()
                noise = torch.normal(0, std, size=weight_copy.size())             
                module.weight.data= torch.exp(noise) * weight_copy
                #print(module)
        elif isinstance(module, nn.Module):
            # If it's a submodule, recursively call the function
            add_noise_to_conv_layers(module, std=std)
            #print(module)

def add_noise_to_conv_layers_pcm(model, yita=0.1):
    for module in model.children():
        if isinstance(module, nn.Conv2d):
            # Add Gaussian noise to convolutional layer weights and biases
            with torch.no_grad():
                weight_copy = module.weight.data.clone() 
                max_value = torch.max(weight_copy)
                std = yita * max_value
                noise = torch.normal(0, std, size=weight_copy.size())             
                module.weight.data= noise + weight_copy
                #print(module)
        elif isinstance(module, nn.Linear):
            # Add Gaussian noise to linear layer weights and biases
            # Add Gaussian noise to convolutional layer weights and biases
            with torch.no_grad():
                weight_copy = module.weight.data.clone() 
                max_value = torch.max(weight_copy)
                std = yita * max_value
                noise = torch.normal(0, std, size=weight_copy.size())             
                module.weight.data= noise + weight_copy
                #print(module)
        elif isinstance(module, nn.Module):
            # If it's a submodule, recursively call the function
            add_noise_to_conv_layers_pcm(module, yita=yita)
            #print(module)

# add_noise_to_conv_layers(net0, std= 0.0*1.5)
# add_noise_to_conv_layers(net1, std= 0.1*1.5)
# add_noise_to_conv_layers(net2, std= 0.2*1.5)
# add_noise_to_conv_layers(net3, std= 0.3*1.5)
# add_noise_to_conv_layers(net4, std= 0.4*1.5)
# add_noise_to_conv_layers(net5, std= 0.5*1.5)

add_noise_to_conv_layers_pcm(net0, yita= 0.00)
add_noise_to_conv_layers_pcm(net1, yita= 0.02)
add_noise_to_conv_layers_pcm(net2, yita= 0.04)
add_noise_to_conv_layers_pcm(net3, yita= 0.06)
add_noise_to_conv_layers_pcm(net4, yita= 0.08)
add_noise_to_conv_layers_pcm(net5, yita= 0.1)

datalist = []

weights0 = net0.state_dict()
for key0 in weights0:
    print(key0)
    if key0 == "conv11.weight":
        datalist.append(weights0[key0].numpy().flatten())
        # plt.hist(weights0[key].numpy().flatten(), range = (-0.15, 0.15), bins=100, alpha=0.1, label="0")

# weights1 = net1.state_dict()
# for key1 in weights1:
#     print(key1)
#     if key1 == "conv11.weight":
#         datalist.append(weights1[key1].numpy().flatten())
#         #plt.hist(weights1[key].numpy().flatten(), range = (-0.15, 0.15), bins=100, alpha=0.1, label="1")

# weights2 = net2.state_dict()
# for key2 in weights2:
#     print(key2)
#     if key2 == "conv11.weight":
#         datalist.append(weights2[key2].numpy().flatten())
#         #plt.hist(weights2[key].numpy().flatten(), range = (-0.15, 0.15), bins=100, alpha=0.1, label="2")

weights3 = net3.state_dict()
for key3 in weights3:
    print(key3)
    if key3 == "conv11.weight":
        datalist.append(weights3[key3].numpy().flatten())
        #plt.hist(weights3[key].numpy().flatten(), range = (-0.15, 0.15), bins=100, alpha=0.1, label="3")

# weights4 = net4.state_dict()
# for key4 in weights4:
#     print(key4)
#     if key4 == "conv11.weight":
#         datalist.append(weights4[key4].numpy().flatten())
#         #plt.hist(weights4[key].numpy().flatten(), range = (-0.15, 0.15), bins=100, alpha=0.1, label="4")

# weights5 = net5.state_dict()
# for key5 in weights5:
#     print(key5)
#     if key5 == "conv11.weight":
#         datalist.append(weights5[key5].numpy().flatten())
#         #plt.hist(weights5[key].numpy().flatten(), range = (-0.15, 0.15), bins=100, alpha=0.1, label="5")

# from scipy.stats import entropy
# p=datalist[0]+1
# q=datalist[1]+1
# p_normalized = p / np.sum(p)
# q_normalized = q / np.sum(q)
 
# # Use scipy.stats.entropy to calculate KL divergence
# kl_divergence = entropy(p_normalized, q_normalized)
# print("---------", kl_divergence)

# Set normal distribution parameters
# mu, sigma = 0, 1
# # Create a sequence of 1000 values generated from normal distribution
# x = np.linspace(mu - 3*sigma, mu + 3*sigma, 1000)
# # Calculate normal distribution probability density values for each x
# pdf = np.exp(-((x - mu)**2 / (2*sigma**2))) / (2*np.pi*sigma**2)
# # Plot probability curve
# plt.plot(x, pdf, c='dodgerblue', label='Probability Density Function')

import pandas as pd
from scipy.stats import norm

for i in range(len(datalist)):
    if i==0:
        plt.hist(datalist[i], range = (-0.03, 0.03), bins=100, alpha=0.6, label="nominal weight")
        np.savetxt('weight_pcm/data.csv', datalist[i], delimiter=',')
        
    if i==1:
        plt.hist(datalist[i], range = (-0.03, 0.03), bins=100, alpha=0.6, label="variational weight") #(\u03c3 = 0.3)
        np.savetxt('weight_pcm/data2.csv', datalist[i], delimiter=',')
       

plt.legend()
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.title('Weight Distribution')
plt.savefig("weight_pcm/plo.png")
plt.show()




net5.eval()
criterion = nn.CrossEntropyLoss()


test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net5(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))







