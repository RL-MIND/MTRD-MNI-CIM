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



parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
opt = parser.parse_args()


rpu_config = InferenceRPUConfig()
rpu_config.forward.is_perfect = False
rpu_config.forward.out_res = -1.  # Turn off (output) ADC discretization.
rpu_config.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
rpu_config.forward.w_noise = 0.00
rpu_config.forward.out_noise = 0.0



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


def main():
    # Build model
    print('Loading model ...\n')
    
    noise_list = [0.0,0.1,0.2,0.3,0.4,0.5]
    for item in noise_list:
        print("@@@@@@@@@@@@@@@@@@@@@@@@@", item)
        for test_i in range(3):
            print("############################", test_i)
            model = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
            model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth')))
            add_noise_to_conv_layers(model, std= item)

            model = convert_to_analog_mapped(model, rpu_config=rpu_config)

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
                print("%s PSNR %f" % (f, psnr))
            psnr_test /= len(files_source)
            print("\nPSNR on test data %f" % psnr_test)

if __name__ == "__main__":
    main()
