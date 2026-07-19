import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import DnCNN, DnCNN_nonid
from dataset import prepare_data, Dataset
from utils import *


parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
parser.add_argument("--w_noiseL", type=float, default=0.0, help='noise level used on validation set')
opt = parser.parse_args()

def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net_t1 = DnCNN_nonid(channels=1, num_of_layers=opt.num_of_layers).cuda()

    net_t2 = DnCNN_nonid(channels=1, num_of_layers=opt.num_of_layers).cuda()

    net_t4 = DnCNN_nonid(channels=1, num_of_layers=opt.num_of_layers).cuda()

    net_t5 = DnCNN_nonid(channels=1, num_of_layers=opt.num_of_layers).cuda()

    src_save1 = opt.outf + "/" + "DnCNN-S-" + str(int(opt.noiseL)) + '/net_0.1.pth'
    checkpoint_t1 = torch.load(src_save1)
    net_t1.load_state_dict(checkpoint_t1)
    net_t1.eval()
    net_t1.train(mode=False)

    src_save2 = opt.outf + "/" + "DnCNN-S-" + str(int(opt.noiseL)) + '/net_0.2.pth'
    checkpoint_t2 = torch.load(src_save2)
    net_t2.load_state_dict(checkpoint_t2)
    net_t2.eval()
    net_t2.train(mode=False)

    src_save4 = opt.outf + "/" + "DnCNN-S-" + str(int(opt.noiseL)) + '/net_0.4.pth'
    checkpoint_t4 = torch.load(src_save4)
    net_t4.load_state_dict(checkpoint_t4)
    net_t4.eval()
    net_t4.train(mode=False)

    src_save5 = opt.outf + "/" + "DnCNN-S-" + str(int(opt.noiseL)) + '/net_0.5.pth'
    checkpoint_t5 = torch.load(src_save5)
    net_t5.load_state_dict(checkpoint_t5)
    net_t5.eval()
    net_t5.train(mode=False)






    model = DnCNN_nonid(channels=1, num_of_layers=opt.num_of_layers).cuda()
    src_save_model = opt.outf + "/" + "DnCNN-S-" + str(int(opt.noiseL)) + '/net_0.3.pth'
    checkpoint_t_model = torch.load(src_save_model)
    model.load_state_dict(checkpoint_t_model)



    criterion = nn.MSELoss(size_average=False)
    def distillation(y, labels, teacher_scores, temp, alpha):
        return nn.KLDivLoss()(torch.log_softmax(y / temp, dim=1), torch.softmax(teacher_scores / temp, dim=1)) * (
                temp * temp * 2.0 * alpha) + criterion(y, labels) * (1. - alpha)
    #criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    writer = SummaryWriter(opt.outf)
    step = 0
    noiseL_B=[0,55] # ingnored when opt.mode=='S'
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data
            if opt.mode == 'S':
                noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL/255.)
            if opt.mode == 'B':
                noise = torch.zeros(img_train.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0,:,:,:].size()
                    noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)
            imgn_train = img_train + noise
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())
            out_train = model(imgn_train, opt.w_noiseL)
            
            outputs_t1 = net_t1(imgn_train, 0.1).detach()
            outputs_t2 = net_t2(imgn_train, 0.2).detach()
            outputs_t4 = net_t4(imgn_train, 0.4).detach()
            outputs_t5 = net_t5(imgn_train, 0.5).detach()
            outputs_t = (outputs_t1+outputs_t2+outputs_t4+outputs_t5)/3.0
            loss = distillation(out_train, noise, outputs_t, temp=5.0, alpha=0.7) / (imgn_train.size()[0]*2)


            #loss = criterion(out_train, noise) / (imgn_train.size()[0]*2)
            loss.backward()
            optimizer.step()
            # results
            model.eval()
            out_train = torch.clamp(imgn_train-model(imgn_train, opt.w_noiseL), 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
        ## the end of each epoch
        model.eval()
        # validate
        psnr_val = 0
        for k in range(len(dataset_val)):
            img_val = torch.unsqueeze(dataset_val[k], 0)
            noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
            imgn_val = img_val + noise
            img_val, imgn_val = Variable(img_val.cuda(), volatile=True), Variable(imgn_val.cuda(), volatile=True)
            out_val = torch.clamp(imgn_val-model(imgn_val, opt.w_noiseL), 0., 1.)
            psnr_val += batch_PSNR(out_val, img_val, 1.)
        psnr_val /= len(dataset_val)
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        # log the images
        out_train = torch.clamp(imgn_train-model(imgn_train, opt.w_noiseL), 0., 1.)
        Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)
        src_save = opt.outf + "/" + "DnCNN-S-" + str(int(opt.noiseL)) + '/net_T_0.1_0.2_0.4_0.5_S_'+str(opt.w_noiseL)+'.pth'
        # save model
        torch.save(model.state_dict(), src_save)

if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=40, stride=10, aug_times=1)
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)
    main()
