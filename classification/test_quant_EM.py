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
from data import MyData

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 test')
parser.add_argument('--precheckpoint_root', default='checkpoint/cifar10/vgg16_nonid/ckpt_cifar10_nonid_T:0.1_0.3_0.5_S:0.3.pth', help='Load precheckpoint', type=str)
parser.add_argument('--noise', default=0.5, type=float, help='noise')
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
    net1 = vgg16(num_classes= num_c)
    net2 = vgg16(num_classes= num_c)
    net3 = vgg16(num_classes= num_c)
    net4 = vgg16(num_classes= num_c)
    net5 = vgg16(num_classes= num_c)
elif args.model_name == "vgg8": #ok
    net1 = vgg8(num_classes= num_c)
    net2 = vgg8(num_classes= num_c)
    net3 = vgg8(num_classes= num_c)
    net4 = vgg8(num_classes= num_c)
    net5 = vgg8(num_classes= num_c)
elif args.model_name == "squeezenet": #ok
    net = SqueezeNet(num_classes= num_c)
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
    net = AlexNet(num_classes= num_c)
elif args.model_name=="vit_small":
    net = ViT( image_size = 32, patch_size = 4, num_classes = 10, dim = int(512), depth = 6, heads = 8, mlp_dim = 512, dropout = 0.1, emb_dropout = 0.1)
else:
    print("Error: model name not define! Exit...")
    exit(1)

net1 = net1.to(device)
net2 = net2.to(device)
net3 = net3.to(device)
net4 = net4.to(device)
net5 = net5.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True



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

noise_list = [0.0,0.1,0.2,0.3,0.4,0.5]
for item in noise_list:
    print("@@@@@@@@@@@@@@@@@@@@@@@@@", item)
    total_acc = 0
    for test_i in range(5):
        print("############################", test_i)
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint1 = torch.load('checkpoint/cifar10/vgg8/ckpt_cifar10_nonid_0.pth')
        checkpoint2 = torch.load('checkpoint/cifar10/vgg8/ckpt_cifar10_kaiming.pth')
        checkpoint3 = torch.load('checkpoint/cifar10/vgg8/ckpt_cifar10_normal.pth')
        checkpoint4 = torch.load('checkpoint/cifar10/vgg8/ckpt_cifar10_uniform.pth')
        checkpoint5 = torch.load('checkpoint/cifar10/vgg8/ckpt_cifar10_xavier.pth')
        net1.load_state_dict(checkpoint1)
        net2.load_state_dict(checkpoint2)
        net3.load_state_dict(checkpoint3)
        net4.load_state_dict(checkpoint4)
        net5.load_state_dict(checkpoint5)

        add_noise_to_conv_layers(net1, item)
        add_noise_to_conv_layers(net2, item)
        add_noise_to_conv_layers(net3, item)
        add_noise_to_conv_layers(net4, item)
        add_noise_to_conv_layers(net5, item)

        net1.eval()
        net2.eval()
        net3.eval()
        net4.eval()
        net5.eval()
        criterion = nn.CrossEntropyLoss()

        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs1 = net1(inputs)
                outputs2 = net2(inputs)
                outputs3 = net3(inputs)
                outputs4 = net5(inputs)
                outputs5 = net5(inputs)
                sum_outputs = outputs1 + outputs2 + outputs3 + outputs4+ outputs5
                outputs = sum_outputs.float()/5
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        total_acc += correct / total
    print("----------total_acc-------", total_acc/5)

        





