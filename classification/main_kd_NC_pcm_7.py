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
from data import MyData

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
best_acc1 = 0  # best test accuracy
best_acc2 = 0  # best test accuracy
best_acc3 = 0  # best test accuracy
best_acc4 = 0  # best test accuracy
best_acc5 = 0  # best test accuracy
acc = 0  # best test accuracy
acc1 = 0  # best test accuracy
acc2 = 0  # best test accuracy
acc3 = 0  # best test accuracy
acc4 = 0  # best test accuracy
acc5 = 0  # best test accuracy
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
    net001 = vgg16_nonid_pcm(num_classes= num_c)
    net002 = vgg16_nonid_pcm(num_classes= num_c)
    net003 = vgg16_nonid_pcm(num_classes= num_c)
    net004 = vgg16_nonid_pcm(num_classes= num_c)
    net005 = vgg16_nonid_pcm(num_classes= num_c)
    net006 = vgg16_nonid_pcm(num_classes= num_c)
    net007 = vgg16_nonid_pcm(num_classes= num_c)
    net008 = vgg16_nonid_pcm(num_classes= num_c)
    net009 = vgg16_nonid_pcm(num_classes= num_c)
    net01 = vgg16_nonid_pcm(num_classes= num_c)
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


net_t001 = net001.to(device)

net_t002 = net002.to(device)

net_t003 = net003.to(device)

net_t004 = net004.to(device)

net_t005 = net005.to(device)

net_t006 = net006.to(device)

net_t007 = net007.to(device)

net_t008 = net008.to(device)

net_t009 = net009.to(device)

net_t01 = net01.to(device)
# checkpoint/cifar100/ResNet18/ckpt_cifar100_nonid_0.0.pth

checkpoint = torch.load('checkpoint_NC/pcm/cifar10/vgg16_nonid_pcm/ckpt_cifar10_nonid_0.06.pth')
net_s.load_state_dict(checkpoint)

checkpoint_t001 = torch.load('checkpoint_NC/pcm/cifar10/vgg16_nonid_pcm/ckpt_cifar10_nonid_0.01.pth')
net_t001.load_state_dict(checkpoint_t001)
net_t001.eval()
net_t001.train(mode=False)

checkpoint_t002 = torch.load('checkpoint_NC/pcm/cifar10/vgg16_nonid_pcm/ckpt_cifar10_nonid_0.02.pth')
net_t002.load_state_dict(checkpoint_t002)
net_t002.eval()
net_t002.train(mode=False)

checkpoint_t003 = torch.load('checkpoint_NC/pcm/cifar10/vgg16_nonid_pcm/ckpt_cifar10_nonid_0.03.pth')
net_t003.load_state_dict(checkpoint_t003)
net_t003.eval()
net_t003.train(mode=False)

checkpoint_t004 = torch.load('checkpoint_NC/pcm/cifar10/vgg16_nonid_pcm/ckpt_cifar10_nonid_0.04.pth')
net_t004.load_state_dict(checkpoint_t004)
net_t004.eval()
net_t004.train(mode=False)

checkpoint_t005 = torch.load('checkpoint_NC/pcm/cifar10/vgg16_nonid_pcm/ckpt_cifar10_nonid_0.05.pth')
net_t005.load_state_dict(checkpoint_t005)
net_t005.eval()
net_t005.train(mode=False)

checkpoint_t006 = torch.load('checkpoint_NC/pcm/cifar10/vgg16_nonid_pcm/ckpt_cifar10_nonid_0.06.pth')
net_t006.load_state_dict(checkpoint_t006)
net_t006.eval()
net_t006.train(mode=False)

checkpoint_t007 = torch.load('checkpoint_NC/pcm/cifar10/vgg16_nonid_pcm/ckpt_cifar10_nonid_0.07.pth')
net_t007.load_state_dict(checkpoint_t007)
net_t007.eval()
net_t007.train(mode=False)

checkpoint_t008 = torch.load('checkpoint_NC/pcm/cifar10/vgg16_nonid_pcm/ckpt_cifar10_nonid_0.08.pth')
net_t008.load_state_dict(checkpoint_t008)
net_t008.eval()
net_t008.train(mode=False)

checkpoint_t009 = torch.load('checkpoint_NC/pcm/cifar10/vgg16_nonid_pcm/ckpt_cifar10_nonid_0.09.pth')
net_t009.load_state_dict(checkpoint_t009)
net_t009.eval()
net_t009.train(mode=False)

checkpoint_t01 = torch.load('checkpoint_NC/pcm/cifar10/vgg16_nonid_pcm/ckpt_cifar10_nonid_0.1.pth')
net_t01.load_state_dict(checkpoint_t01)
net_t01.eval()
net_t01.train(mode=False)


criterion = nn.CrossEntropyLoss()


optimizer = optim.SGD(net_s.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

def distillation(y, labels, teacher_scores, temp, alpha):
    return nn.KLDivLoss()(torch.log_softmax(y / temp, dim=1), torch.softmax(teacher_scores / temp, dim=1)) * (
            temp * temp * 2.0 * alpha) + criterion(y, labels) * (1. - alpha)

# Training
def train(epoch,a1=1,a2=1,a4=1,a5=1):
    print("a1", a1)
    print("a2", a2)
    print("a4", a4)
    print("a5", a5)
    print('\nEpoch: %d' % epoch)
    net_s.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # print(inputs.shape)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net_s(inputs, args.noise)
        #outputs_t001 = net_t001(inputs, 0.01).detach()
        outputs_t002 = net_t002(inputs, 0.02).detach()
        #outputs_t003 = net_t003(inputs, 0.03).detach()
        #outputs_t004 = net_t004(inputs, 0.04).detach()
        #outputs_t005 = net_t005(inputs, 0.05).detach()
        #outputs_t006 = net_t006(inputs, 0.06).detach()
        #outputs_t007 = net_t007(inputs, 0.07).detach()
        #outputs_t008 = net_t008(inputs, 0.08).detach()
        #outputs_t009 = net_t009(inputs, 0.09).detach()
        #outputs_t01 = net_t01(inputs, 0.1).detach()
        #outputs_t3 = net_t3(inputs, lamda=0.5).detach()
        #out_a1 = outputs_t002
        # out_a2 = outputs_t004
        # out_a4 = outputs_t008
        #out_a5 = outputs_t01
        outputs_t = outputs_t002
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




def test1(epoch):
    global acc1
    net_s.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net_s(inputs, lamda=0.02)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    acc1 = 100. * correct / total
    return acc1

def test2(epoch):
    global acc2
    net_s.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net_s(inputs, lamda=0.04)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    acc2 = 100. * correct / total
    return acc2

def test3(epoch):
    global acc3
    net_s.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net_s(inputs, lamda=0.06)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    acc3 = 100. * correct / total
    return acc3

def test4(epoch):
    global acc4
    net_s.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net_s(inputs, lamda=0.08)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    acc4 = 100. * correct / total
    return acc4

def test5(epoch):
    global acc5
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
    acc5 = 100. * correct / total
    return acc5
def sort_list_with_index(input_list):
    sorted_list = sorted(enumerate(input_list), key=lambda x: x[1])
    return [x[0] for x in sorted_list]

def test_all(epoch):
    global best_acc1
    global best_acc2
    global best_acc3
    global best_acc4
    global best_acc5
    acc1 = test1(epoch)
    acc2 = test2(epoch)
    acc3 = test3(epoch)
    acc4 = test4(epoch)
    acc5 = test5(epoch)

    if not os.path.isdir('checkpoint/'+args.dataset+'/'+args.model_name):
            os.mkdir('checkpoint/'+args.dataset+'/'+args.model_name)
    torch.save(net_s.state_dict(), "checkpoint_NC/pcm/cifar10/vgg16_nonid_pcm/"+'/ckpt_'+args.dataset+'_nonid_T:0.02_bh7_S:'+str(args.noise)+'.pth')
    list_acc = [acc1, acc2, acc4, acc5]
    list_beat_acc = [best_acc1, best_acc2, best_acc4, best_acc5]
    differences = [x - y for x, y in zip(list_acc, list_beat_acc)]
    sorted_index = sort_list_with_index(differences)
    list_w = [0,0,0,0]
    list_w[sorted_index[0]]=2.5
    list_w[sorted_index[1]]=2
    list_w[sorted_index[2]]=1.5
    list_w[sorted_index[3]]=1
    best_acc1 = acc1
    best_acc2 = acc2
    best_acc3 = acc3
    best_acc4 = acc4
    best_acc5 = acc5

    return  list_w[0], list_w[1], list_w[2], list_w[3]




    
a1 =1
a2 =1
a4 =1
a5 =1
for epoch in range(start_epoch, start_epoch + args.epoch):
    train(epoch, a1, a2, a4, a5)
    a1, a2, a4, a5 = test_all(epoch)
    scheduler.step()
