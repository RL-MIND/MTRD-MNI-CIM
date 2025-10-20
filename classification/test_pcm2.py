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

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, ConstantStepDevice, FloatingPointRPUConfig, FloatingPointDevice
from aihwkit.simulator.rpu_base import cuda
from aihwkit.nn.conversion import convert_to_analog_mapped

from aihwkit.nn.conversion import convert_to_analog_mapped
from aihwkit.simulator.configs import SingleRPUConfig, ConstantStepDevice
from aihwkit.utils.analog_info import analog_summary
from aihwkit.simulator.configs import (
    InferenceRPUConfig,
    WeightNoiseType,
    WeightClipType,
    WeightModifierType,
)
from aihwkit.inference import PCMLikeNoiseModel

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 test')
parser.add_argument('--precheckpoint_root', default='checkpoint/cifar100/ResNet18/ckpt_cifar100_nonid_0.0.pth', help='Load precheckpoint', type=str)
parser.add_argument('--noise', default=0.5, type=float, help='noise')
parser.add_argument('--quant_bit', default=8, type=int, help='quant')
parser.add_argument('--model_name', default='ResNet18', help='choice a model to train and eval. eg: alenet, vgg16', type=str)
parser.add_argument('--data_root', default='data/cifar10_dataset', help='Path to the train dataset', type=str)
parser.add_argument('--num_workers', default=0, help='number of workers', type=int)
parser.add_argument('--batch_size', default=64, help='number of batch size', type=int)
parser.add_argument('--dataset', type=str, default='cifar100',help='training dataset (default: cifar100)')
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
    net = vgg16(num_classes= num_c)
elif args.model_name == "vgg8": #ok
    net = vgg8(num_classes= num_c)
elif args.model_name == "squeezenet": #ok
    net = SqueezeNet(num_classes= num_c)
elif args.model_name == "ResNet18": #ok
    net = ResNet18(num_classes= num_c)
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

net = net.to(device)
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

# Define a single-layer network, using inference/hardware-aware training tile
rpu_config = InferenceRPUConfig()
rpu_config.forward.is_perfect = False
rpu_config.forward.out_res = -1.  # Turn off (output) ADC discretization.
rpu_config.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
rpu_config.forward.w_noise = 0.00
rpu_config.forward.out_noise = 0.00

noise_list = [0.0,0.1,0.2,0.3,0.4,0.5]
for item in noise_list:
    print("@@@@@@@@@@@@@@@@@@@@@@@@@", item)
    for test_i in range(3):
        print("############################", test_i)
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.precheckpoint_root)
        net.load_state_dict(checkpoint)

        add_noise_to_conv_layers(net, std= item)

        net.eval()
        criterion = nn.CrossEntropyLoss()

        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        net_pcm = convert_to_analog_mapped(net, rpu_config=rpu_config)
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net_pcm(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))







                





