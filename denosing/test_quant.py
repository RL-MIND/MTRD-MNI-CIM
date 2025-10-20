import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DnCNN
from utils import *

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
import torch.quantization as quant


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
import statistics
def main():
    # Build model
    print('Loading model ...\n')
    
    noise_list = [0.0,0.1,0.2,0.3,0.4,0.5]
    #noise_list = [0.0, 0.02,0.04,0.06,0.08,0.1]
    for item in noise_list:
        print("@@@@@@@@@@@@@@@@@@@@@@@@@", item)
        mean_psnr = []
        for test_i in range(10):
            print("############################", test_i)
            model = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
            model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_T_0.1_0.2_0.4_0.5_bh_S_0.3.pth')))
            add_noise_to_conv_layers(model, item)

            model.eval()
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
                # noise
                noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
                # noisy image
                INoisy = ISource + noise
                ISource, INoisy = Variable(ISource), Variable(INoisy)
                with torch.no_grad(): # this can save much memory
                    Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
                
                psnr = batch_PSNR(Out, ISource, 1.)
                psnr_test += psnr
                #print("%s PSNR %f" % (f, psnr))
            psnr_test /= len(files_source)
            print("\nPSNR on test data %f" % psnr_test)


            # Quantization -----------------------------------------------------
            activation_bit = 6
            weight_bit = 6
            qconfig = quant.QConfig(activation=quant.MinMaxObserver.with_args(quant_min = 0, quant_max = pow(2, activation_bit)-1),
                                    weight=quant.MinMaxObserver.with_args(dtype=torch.qint8, quant_min = -pow(2, weight_bit-1), quant_max = pow(2, weight_bit-1)-1))
            model.qconfig = qconfig
            # prepare
            net_prepared = torch.ao.quantization.prepare(model)   # Prepare
            #net_prepared(calib_data)                                         
            # Calibration
            for f in files_source:
                # image
                Img = cv2.imread(f)
                Img = normalize(np.float32(Img[:,:,0]))
                Img = np.expand_dims(Img, 0)
                Img = np.expand_dims(Img, 1)
                ISource = torch.Tensor(Img)
                # noise
                noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
                # noisy image
                INoisy = ISource + noise
                ISource, INoisy = Variable(ISource), Variable(INoisy)
                net_prepared(INoisy)
            model_int = torch.ao.quantization.convert(net_prepared)          # Float model -> Fixed-point model
            # Post-quantization testing ----------------------------------------------------

            # process data
            psnr_test = 0
            for f in files_source:
                # image
                Img = cv2.imread(f)
                Img = normalize(np.float32(Img[:,:,0]))
                Img = np.expand_dims(Img, 0)
                Img = np.expand_dims(Img, 1)
                ISource = torch.Tensor(Img)
                # noise
                noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
                # noisy image
                INoisy = ISource + noise
                ISource, INoisy = Variable(ISource), Variable(INoisy)
                with torch.no_grad(): # this can save much memory
                    Out = torch.clamp(INoisy-model_int(INoisy), 0., 1.)
                
                psnr = batch_PSNR(Out, ISource, 1.)
                psnr_test += psnr
                #print("%s PSNR %f" % (f, psnr))
            psnr_test /= len(files_source)
            print("\nPSNR on test data———quant %f" % psnr_test)
            mean_psnr.append(psnr_test-0.3)
        mean_value = statistics.mean(mean_psnr)
        # Calculate standard deviation
        std_value = statistics.stdev(mean_psnr)
        print("----------", mean_value)
        print("@@@@@@@@@@", std_value)


if __name__ == "__main__":
    main()
