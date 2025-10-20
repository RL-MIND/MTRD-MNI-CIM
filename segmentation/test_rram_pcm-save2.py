from pkgutil import get_loader
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET, UNET_nonid, UNET_nonid_pcm
import torch.quantization as quant
import numpy

from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)


LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
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

def add_noise_to_conv_layers(model, std=0.1):
    for module in model.children():
        if isinstance(module, nn.Conv2d):
            # Add Gaussian noise to convolutional layer weights and biases
            with torch.no_grad():
                weight_copy = module.weight.data.clone()
                noise = torch.normal(0, std, size=weight_copy.size())             
                module.weight.data= torch.exp(noise.to(DEVICE)) * weight_copy
                #print(module)
        elif isinstance(module, nn.Linear):
            # Add Gaussian noise to linear layer weights and biases
            with torch.no_grad():
                weight_copy = module.weight.data.clone() 
                noise = torch.normal(0, std, size=weight_copy.size())             
                module.weight.data= torch.exp(noise.to(DEVICE)) * weight_copy
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
                module.weight.data= noise.to(DEVICE) + weight_copy
                #print(module)
        elif isinstance(module, nn.Linear):
            # Add Gaussian noise to linear layer weights and biases
            with torch.no_grad():
                weight_copy = module.weight.data.clone() 
                max_value = torch.max(weight_copy)
                std = yita * max_value
                noise = torch.normal(0, std, size=weight_copy.size())             
                module.weight.data= noise.to(DEVICE) + weight_copy
                #print(module)
        elif isinstance(module, nn.Module):
            # If it's a submodule, recursively call the function
            add_noise_to_conv_layers_pcm(module, yita=yita)
            #print(module)

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

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
    import statistics
    #noise_list = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2]
    #noise_list = [0.02, 0.04, 0.06, 0.08, 0.1]
    noise_list = [0.1,0.2,0.3,0.4,0.5]
    for item in noise_list:
        print("@@@@@@@@@@@@@@@@@@@@@@@@@", item)
        all_acc = []
        all_dice = []
        for test_i in range(1):
            print("############################", test_i)

            model = UNET(in_channels=3, out_channels=1).to(DEVICE)
            
            load_checkpoint(torch.load("0.0my_checkpoint.pth.tar"), model)
            
            add_noise_to_conv_layers(model, item)
           

            check_accuracy(val_loader, model, lamda=item, device=DEVICE)
            save_folder = "save_nominal/rram/"+str(item)+"/"
            save_predictions_as_imgs(val_loader, model, folder=save_folder, device=DEVICE)
            #all_dice.append((dice.cpu()).numpy())
        # mean_value = statistics.mean(all_dice)
        # # Calculate standard deviation
        # std_value = statistics.stdev(all_dice)
        # print("----------", mean_value)
        # print("@@@@@@@@@@", std_value)

    
        

if __name__ == "__main__":
    main()




