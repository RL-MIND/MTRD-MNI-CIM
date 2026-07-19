'''Train CIFAR10 with PyTorch.'''
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import os
import argparse
from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
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

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
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
if args.model_name == "VGG16":
    net = vgg16()
elif args.model_name == "vgg16_nonid": #ok
    net = vgg16_nonid()
elif args.model_name == "vgg8": #ok
    net = vgg8(num_classes= 10)
elif args.model_name == "squeezenet": #ok
    net = squeezenet()
elif args.model_name == "VGG11": #ok
    net = VGG('VGG11')
elif args.model_name == "VGG13": #ok
    net = VGG('VGG13')
elif args.model_name == "VGG19": #ok
    net = VGG('VGG19')
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

net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.precheckpoint_root)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()


optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epoch)

def b_initialize_weights(module):
    for m in module():
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

def uniform_initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.uniform_(m.weight.data)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
def normal_initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, mean=0, std=1)
            if m.bias is not None:
                nn.init.normal_(m.bias.data, mean=0, std=1)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, mean=0, std=1)
            nn.init.normal_(m.bias.data, mean=0, std=1)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, mean=0, std=1)
            if m.bias is not None:
                nn.init.normal_(m.bias.data, mean=0, std=1)
def kaiming_initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)

def xavier_initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)

def trunc_initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight.data)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight.data)

def zero_initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.zeros_(m.weight.data)
        elif isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight.data)
def kaiming_uniform_initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform(m.weight.data)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform(m.weight.data)
def xavier_uniform_initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight.data)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight.data)
xavier_uniform_initialize_weights(net)



# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # print(inputs.shape)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        #print('outputs',outputs)
        #print('targets',targets)
        loss = criterion(outputs, targets)
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
    net.eval()
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

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint/'+args.model_name):
            os.mkdir('checkpoint/'+args.model_name)
        torch.save(net.state_dict(), 'checkpoint/'+str(args.dataset)+'/'+args.model_name+'/ckpt_'+args.dataset+'_xavier_uniform.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch + args.epoch):
    train(epoch)
    test(epoch)
    scheduler.step()
