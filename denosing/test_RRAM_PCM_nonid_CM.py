import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DnCNN,DnCNN_smi
from utils import *
import torch.quantization as quant
# python test_RRAM_PCM_nonid_CM.py --num_of_layers 17 --logdir logs/DnCNN-S-25 --test_data Set12 --test_noiseL 25
parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
opt = parser.parse_args()

def normalize(data):
    return data/255.
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
def main():
    # Build model
    import statistics
    #noise_list = [0.0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16]
    noise_list = [0.1,0.2,0.3,0.4,0.5]
    for item in noise_list:
        print("@@@@@@@@@@@@@@@@@@@@@@@@@", item)
        mean_psnr = []
        for test_i in range(5):
            print("############################", test_i)
            print('Loading model ...\n')
            model_1 = DnCNN_smi(channels=1, num_of_layers=opt.num_of_layers)
            model_2 = DnCNN_smi(channels=1, num_of_layers=opt.num_of_layers)
            model_3 = DnCNN_smi(channels=1, num_of_layers=opt.num_of_layers)
            model_4 = DnCNN_smi(channels=1, num_of_layers=opt.num_of_layers)
            model_5 = DnCNN_smi(channels=1, num_of_layers=opt.num_of_layers)
            model_1.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_kaiming_initialize.pth')))
            model_2.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_kaiming_uniform.pth')))
            model_3.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_xavier_initialize.pth')))
            model_4.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_xavier_uniform.pth')))
            model_5.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_normal.pth')))
            add_noise_to_conv_layers(model_1, item)
            add_noise_to_conv_layers(model_2, item)
            add_noise_to_conv_layers(model_3, item)
            add_noise_to_conv_layers(model_4, item)
            add_noise_to_conv_layers(model_5, item)
            model_1.eval()
            model_2.eval()
            model_3.eval()
            model_4.eval()
            model_5.eval()
            # load data info
            print('Loading data info ...\n')
            files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
            files_source.sort()

            

            # process data
            psnr_test = 0
            for f in files_source:
                # image
                Img = cv2.imread(f)
                Img = normalize(np.float32(Img[:,:,0]))
                Img = np.expand_dims(Img, 0)
                Img = np.expand_dims(Img, 1)
                ISource = torch.Tensor(Img)
                print(ISource.shape)
                # noise
                noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
                # noisy image
                INoisy = ISource + noise
                ISource, INoisy = Variable(ISource), Variable(INoisy)
                with torch.no_grad(): # this can save much memory
                    Out_1 = torch.clamp(INoisy-model_1(INoisy), 0., 1.)
                    Out_2 = torch.clamp(INoisy-model_2(INoisy), 0., 1.)
                    Out_3 = torch.clamp(INoisy-model_3(INoisy), 0., 1.)
                    Out_4 = torch.clamp(INoisy-model_4(INoisy), 0., 1.)
                    Out_5 = torch.clamp(INoisy-model_5(INoisy), 0., 1.)
                ## if you are using older version of PyTorch, torch.no_grad() may not be supported
                # ISource, INoisy = Variable(ISource.cuda(),volatile=True), Variable(INoisy.cuda(),volatile=True)
                # Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
                Out = (Out_1 + Out_2 + Out_3 + Out_4 + Out_5)/5
                psnr = batch_PSNR(Out, ISource, 1.)
                psnr_test += psnr
                #print("%s PSNR %f" % (f, psnr))
            psnr_test /= len(files_source)
            print("\nPSNR on test data %f" % psnr_test)
            mean_psnr.append(psnr_test)
        mean_value = statistics.mean(mean_psnr)
        # Calculate standard deviation
        std_value = statistics.stdev(mean_psnr)
        print("----------", mean_value)
        print("@@@@@@@@@@", std_value)

if __name__ == "__main__":
    main()
