from pkgutil import get_loader
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET, UNET_nonid_src,UNET_nonid

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
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)


LEARNING_RATE = 1e-4
DEVICE = "cpu"
BATCH_SIZE = 16
NUM_EPOCHES = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "data/imgs/train_hq/"
TRAIN_MASK_DIR = "data/masks/train_masks/"
VAL_IMG_DIR = "data/imgs/test_hq"
VAL_MASK_DIR = "data/masks/test_masks/"


rpu_config = InferenceRPUConfig()
rpu_config.forward.is_perfect = False
rpu_config.forward.out_res = -1.  # Turn off (output) ADC discretization.
rpu_config.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
rpu_config.forward.w_noise = 0.00
rpu_config.forward.out_noise = 0.00
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
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0,0.0,0.0],
                std=[1.0,1.0,1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    #model = UNET(in_channels=3, out_channels=1).to(DEVICE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    noise_list = [0.0,0.1,0.2,0.3,0.4,0.5]
    for item in noise_list:
        print("@@@@@@@@@@@@@@@@@@@@@@@@@", item)
        for test_i in range(3):
            print("############################", test_i)
            model = UNET(in_channels=3, out_channels=1).to(DEVICE)
            if LOAD_MODEL:
                load_checkpoint(torch.load("0.0my_checkpoint.pth.tar"), model)
            add_noise_to_conv_layers(model, std= item)

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
            for x, y in val_loader:
                x = x.to(DEVICE)
                net_prepared(x)
            model_int = torch.ao.quantization.convert(net_prepared)          # Float model -> Fixed-point model
            # Post-quantization testing ----------------------------------------------------

            check_accuracy(val_loader, model_int, device=DEVICE)
    
        

if __name__ == "__main__":
    main()




