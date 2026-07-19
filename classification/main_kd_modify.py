'''Train CIFAR10 with PyTorch using Knowledge Distillation.

使用方法:
1. 指定教师数量: --num_teachers 3
2. 指定教师模型路径: --teacher_paths path1.pth path2.pth path3.pth
3. 指定学生模型路径: --student_path student.pth

示例:
python main_kd.py --model_name squeezenet_nonid --num_teachers 2 \
    --teacher_paths checkpoint/cifar10/squeezenet/ckpt1.pth checkpoint/cifar10/squeezenet/ckpt2.pth \
    --student_path checkpoint/cifar10/squeezenet/student.pth
'''
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import os
import argparse
from models import *
from utils import progress_bar

def create_model(model_name, num_classes):
    """创建指定的模型"""
    if model_name == "vgg16":
        return vgg16(num_classes=num_classes)
    elif model_name == "vgg16_nonid":
        return vgg16_nonid(num_classes=num_classes)
    elif model_name == "vgg8":
        return vgg8(num_classes=num_classes)
    elif model_name == "vgg8_nonid":
        return vgg8_nonid(num_classes=num_classes)
    elif model_name == "squeezenet":
        return SqueezeNet(num_classes=num_classes)
    elif model_name == "squeezenet_nonid":
        return SqueezeNet_nonid(num_classes=num_classes)
    elif model_name == "ResNet18":
        return ResNet18()
    elif model_name == "ResNet34":
        return ResNet34()
    elif model_name == "ResNet50":
        return ResNet50()
    elif model_name == "PreActResNet18":
        return PreActResNet18()
    elif model_name == "PreActResNet34":
        return PreActResNet34()
    elif model_name == "PreActResNet50":
        return PreActResNet50()
    elif model_name == "PreActResNet101":
        return PreActResNet101()
    elif model_name == "PreActResNet152":
        return PreActResNet152()
    elif model_name == "GoogLeNet":
        return GoogLeNet()
    elif model_name == "DenseNet121":
        return DenseNet121()
    elif model_name == "ResNeXt29_2x64d":
        return ResNeXt29_2x64d()
    elif model_name == "MobileNet":
        return MobileNet()
    elif model_name == "MobileNetV2":
        return MobileNetV2()
    elif model_name == "DPN92":
        return DPN92()
    elif model_name == "ShuffleNetG2":
        return ShuffleNetG2()
    elif model_name == "SENet18":
        return SENet18()
    elif model_name == "ShuffleNetV2":
        return ShuffleNetV2(1)
    elif model_name == "EfficientNetB0":
        return EfficientNetB0()
    elif model_name == "RegNetX_200MF":
        return RegNetX_200MF()
    elif model_name == "SimpleDLA":
        return SimpleDLA()
    elif model_name == "LeNet":
        return LeNet()
    elif model_name == "AlexNet":
        return AlexNet(num_classes=num_classes)
    elif model_name == "AlexNet_nonid":
        return AlexNet_nonid(num_classes=num_classes)
    elif model_name == "vit_small":
        return ViT(image_size=32, patch_size=4, num_classes=10, dim=int(512), depth=6, heads=8, mlp_dim=512, dropout=0.1, emb_dropout=0.1)
    else:
        print("Error: model name not define! Exit...")
        exit(1)


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
parser.add_argument('--num_teachers', default=3, type=int, help='number of teacher models')
parser.add_argument('--teacher_paths', nargs='+', help='paths to teacher model checkpoints', required=False)
parser.add_argument('--student_path', help='path to student model checkpoint', type=str, required=False)

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
net = create_model(args.model_name, num_c)


net_s = net.to(device)

# 动态创建教师模型
print(f'==> Creating {args.num_teachers} teacher models..')
teacher_nets = []
for i in range(args.num_teachers):
    # 创建教师模型（与学生模型相同的架构）
    teacher_net = create_model(args.model_name, num_c)
    teacher_net = teacher_net.to(device)
    teacher_nets.append(teacher_net)

# 加载学生模型权重
if args.student_path:
    print(f'==> Loading student model from {args.student_path}')
    checkpoint = torch.load(args.student_path)
    net_s.load_state_dict(checkpoint)
else:
    # 默认路径（保持向后兼容）
    checkpoint = torch.load('checkpoint/cifar10/squeezenet/ckpt_cifar10_nonid_0.3.pth')
    net_s.load_state_dict(checkpoint)

# 加载教师模型权重
if args.teacher_paths:
    if len(args.teacher_paths) != args.num_teachers:
        print(f"Error: Number of teacher paths ({len(args.teacher_paths)}) does not match number of teachers ({args.num_teachers})")
        exit(1)
    
    for i, teacher_path in enumerate(args.teacher_paths):
        print(f'==> Loading teacher model {i+1} from {teacher_path}')
        checkpoint_t = torch.load(teacher_path)
        teacher_nets[i].load_state_dict(checkpoint_t)
        teacher_nets[i].eval()
        teacher_nets[i].train(mode=False)
else:
    # 默认路径（保持向后兼容）
    default_paths = [
        'checkpoint/cifar10/squeezenet/ckpt_cifar10_nonid_0.1.pth',
        'checkpoint/cifar10/squeezenet/ckpt_cifar10_nonid_0.3.pth',
        'checkpoint/cifar10/squeezenet/ckpt_cifar10_nonid_0.5.pth'
    ]
    
    for i in range(min(args.num_teachers, len(default_paths))):
        print(f'==> Loading teacher model {i+1} from {default_paths[i]}')
        checkpoint_t = torch.load(default_paths[i])
        teacher_nets[i].load_state_dict(checkpoint_t)
        teacher_nets[i].eval()
        teacher_nets[i].train(mode=False)

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
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

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
        
        # 获取所有教师模型的输出
        teacher_outputs = []
        for teacher_net in teacher_nets:
            teacher_output = teacher_net(inputs, lamda=args.noise).detach()
            teacher_outputs.append(teacher_output)
        
        # 计算教师模型输出的平均值
        if len(teacher_outputs) > 0:
            outputs_t = sum(teacher_outputs) / len(teacher_outputs)
        else:
            outputs_t = outputs  # 如果没有教师模型，使用学生模型自己的输出
        
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
        torch.save(net_s.state_dict(), 'checkpoint/'+args.dataset+'/'+args.model_name+'/ckpt_'+args.dataset+'_nonid_T:'+str(args.num_teachers)+'_S:'+str(args.noise)+'.pth')
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
