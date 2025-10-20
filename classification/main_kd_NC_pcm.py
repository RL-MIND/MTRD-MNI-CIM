'''Train CIFAR10 with PyTorch.'''
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import torch.nn.functional as F
import os
import argparse

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--noise', default=0.0, type=float, help='noise')
parser.add_argument('--epoch', default=100, type=int, help='learning epoch')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--precheckpoint_root', help='Load precheckpoint', type=str)
parser.add_argument("--optim", default="SGD", help='optim', type=str)
parser.add_argument('--model_name', help='choice a model to train and eval. eg: alenet, vgg16', type=str)
parser.add_argument('--data_root', help='Path to the train dataset', type=str)
parser.add_argument('--batch_size', default=64, help='number of batch size', type=int)
parser.add_argument('--num_workers', default=8, help='number of workers', type=int)
parser.add_argument('--dataset', type=str, default='cifar10',help='training dataset (default: cifar100)')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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
elif args.model_name == "vgg16_nonid": #ok
    nets = vgg16_nonid_pcm(num_classes= num_c)
    net1 = vgg16_nonid_pcm(num_classes= num_c)
    net2 = vgg16_nonid_pcm(num_classes= num_c)
    net3 = vgg16_nonid_pcm(num_classes= num_c)
    net4 = vgg16_nonid_pcm(num_classes= num_c)
    net5 = vgg16_nonid_pcm(num_classes= num_c)
elif args.model_name == "vgg8": #ok
    net = vgg8(num_classes= num_c)
elif args.model_name == "vgg8_nonid": #ok
    net = vgg8_nonid(num_classes= num_c)
elif args.model_name == "squeezenet": #ok
    net = SqueezeNet(num_classes= num_c)
elif args.model_name == "squeezenet_nonid": #ok
    net = SqueezeNet_nonid(num_classes= num_c)
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
elif args.model_name == "AlexNet_nonid":  #ok
    net = AlexNet_nonid(num_classes= num_c)
elif args.model_name=="vit_small":
    net = ViT( image_size = 32, patch_size = 4, num_classes = 10, dim = int(512), depth = 6, heads = 8, mlp_dim = 512, dropout = 0.1, emb_dropout = 0.1)
else:
    print("Error: model name not define! Exit...")
    exit(1)


net_s = nets.to(device)


net_t1 = net1.to(device)

net_t2 = net2.to(device)

net_t3 = net3.to(device)

net_t4 = net4.to(device)

net_t5 = net5.to(device)
# checkpoint/cifar100/ResNet18/ckpt_cifar100_nonid_0.0.pth

checkpoint = torch.load('checkpoint_NC/pcm/cifar10/vgg16_nonid_pcm/ckpt_cifar10_nonid_0.06.pth')
net_s.load_state_dict(checkpoint)

checkpoint_t1 = torch.load('checkpoint_NC/pcm/cifar10/vgg16_nonid_pcm/ckpt_cifar10_nonid_0.02.pth')
net_t1.load_state_dict(checkpoint_t1)
net_t1.eval()
net_t1.train(mode=False)

checkpoint_t2 = torch.load('checkpoint_NC/pcm/cifar10/vgg16_nonid_pcm/ckpt_cifar10_nonid_0.04.pth')
net_t2.load_state_dict(checkpoint_t2)
net_t2.eval()
net_t2.train(mode=False)

checkpoint_t3 = torch.load('checkpoint_NC/pcm/cifar10/vgg16_nonid_pcm/ckpt_cifar10_nonid_0.06.pth')
net_t3.load_state_dict(checkpoint_t3)
net_t3.eval()
net_t3.train(mode=False)

checkpoint_t4 = torch.load('checkpoint_NC/pcm/cifar10/vgg16_nonid_pcm/ckpt_cifar10_nonid_0.08.pth')
net_t4.load_state_dict(checkpoint_t4)
net_t4.eval()
net_t4.train(mode=False)

checkpoint_t5 = torch.load('checkpoint_NC/pcm/cifar10/vgg16_nonid_pcm/ckpt_cifar10_nonid_0.1.pth')
net_t5.load_state_dict(checkpoint_t5)
net_t5.eval()
net_t5.train(mode=False)

# checkpoint_t3 = torch.load('checkpoint/vgg16_nonid/ckpt_cifar10_nonid_0.5.pth')
# net_t3.load_state_dict(checkpoint_t3)
# net_t3.eval()
# net_t3.train(mode=False)

# if args.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load(args.precheckpoint_root)
#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()


optimizer = optim.SGD(net_s.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

def distillation(y, labels, teacher_scores, temp, alpha):
    return nn.KLDivLoss()(torch.log_softmax(y / temp, dim=1), torch.softmax(teacher_scores / temp, dim=1)) * (
            temp * temp * 2.0 * alpha) + criterion(y, labels) * (1. - alpha)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net_s.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # print(inputs.shape)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net_s(inputs, lamda=args.noise)
        #outputs_t1 = net_t1(inputs, lamda=0.02).detach()
        outputs_t2 = net_t2(inputs, lamda=0.04).detach()
        #outputs_t3 = net_t2(inputs, lamda=0.06).detach()
        outputs_t4 = net_t4(inputs, lamda=0.08).detach()
        #outputs_t5 = net_t5(inputs, lamda=0.1).detach()
        outputs_t = (outputs_t2+outputs_t4)/2.0
        loss = distillation(outputs, targets, outputs_t, temp=5.0, alpha=0.7)
        #print('outputs',outputs)
        #print('targets',targets)
        #loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(epoch):
    global best_acc
    net_s.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net_s(inputs, lamda=args.noise)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net_s.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint/'+args.dataset+'/'+args.model_name):
            os.mkdir('checkpoint/'+args.dataset+'/'+args.model_name)
        torch.save(net_s.state_dict(), "checkpoint_NC/pcm/cifar10/vgg16_nonid_pcm/"+'/ckpt_'+args.dataset+'_nonid_T:0.04_0.08_S:'+str(args.noise)+'.pth')
        best_acc = acc

def test1(epoch):
    global best_acc
    net_s.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net_s(inputs, lamda=0.1)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

def test2(epoch):
    global best_acc
    net_s.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net_s(inputs, lamda=0.5)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

for epoch in range(start_epoch, start_epoch + args.epoch):
    train(epoch)
    test(epoch)
    test1(epoch)
    test2(epoch)
    scheduler.step()
