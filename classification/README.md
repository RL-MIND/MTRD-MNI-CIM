# MTRD-MNI-CIM Classification 

This repository contains the classification experiments for MTRD-MNI-CIM research, now with improved code organization and modularity.

## Overview

This project focuses on neural network classification tasks with emphasis on:
- CIFAR-10/CIFAR-100 dataset classification
- Various neural network architectures (VGG, ResNet, MobileNet, etc.)
- Noise robustness analysis (RRAM, PCM, Gaussian)
- Knowledge distillation techniques
- Quantization effects on model performance
- Memory device simulation (ReRAM, PCM)

## Requirements

- Python 3.7+
- PyTorch 1.8+
- torchvision
- numpy
- matplotlib
- tqdm

## Installation

```bash
pip install torch torchvision numpy matplotlib tqdm
```

## Usage

### 1. Basic Training for teacher model

```bash
# modify you want to training noise level parameter
python main_NC.py --model_name vgg16 --dataset cifar10 --epochs 200 --lr 0.01 --noise 0.0
```

### 2. Knowledge Distillation

```bash
# configure teacher number and model path on the main_kd.py
python main_kd.py --model_name vgg16 --dataset cifar10 
```

### 3. Weight Analysis

```bash
# You can imitate and modify
python weight_print_weight_add.py
python weight_print_weight.py
python weight_print.py
python weight_print2.py
python weight_print2_plo.py
python weight_print2_plo2.py
```

### Legacy Training 

```bash
# Start training with: 
python main.py

# You can manually resume the training with: 
python main.py --resume --lr=0.01
```
## Model Architectures

Supported models:
- **VGG**: vgg8, vgg11, vgg13, vgg16, vgg19, vgg16_nonid
- **ResNet**: resnet18, resnet34, resnet50, resnet101, resnet152
- **PreAct ResNet**: preactresnet18, preactresnet34, preactresnet50
- **MobileNet**: mobilenet, mobilenetv2
- **DenseNet**: densenet121, densenet169, densenet201
- **Others**: squeezenet, googlenet, shufflenet, senet18, efficientnet, regnety_400mf

## Datasets

- CIFAR-10: 10 classes, 32x32 RGB images
- CIFAR-100: 100 classes, 32x32 RGB images

## Noise Simulation

### Supported Noise Types
- **RRAM**: Resistive RAM noise simulation
- **PCM**: Phase Change Memory noise simulation  
- **Gaussian**: Standard Gaussian noise

### Noise Parameters
- `noise_std`: Standard deviation of noise
- `noise_yita`: PCM-specific noise factor


- **VGG16**
```
train: noise:0
test: noise:0(93.73); noise:0.1(93.29); noise:0.2(92.41); noise:0.3(89.95); noise:0.4(83.15); noise:0.5(66.79);noise:0.6(36.26)

train: noise:0.1
test: noise:0(93.79); noise:0.1(93.51); noise:0.2(92.60); noise:0.3(91.13); noise:0.4(85.94); noise:0.5(74.24);noise:0.6(47.69)

train: noise:0.2
test: noise:0(93.62); noise:0.1(93.46); noise:0.2(93.02); noise:0.3(92.02); noise:0.4(89.53); noise:0.5(81.23);noise:0.6(59.85)

train: noise:0.3
test: noise:0(92.69); noise:0.1(92.68); noise:0.2(92.48); noise:0.3(92.10); noise:0.4(90.28); noise:0.5(84.71);noise:0.6(68.39)

train: noise:0.4
test: noise:0(90.62); noise:0.1(91.00); noise:0.2(91.48); noise:0.3(91.70); noise:0.4(91.30); noise:0.5(88.39);noise:0.6(78.74)

train: noise:0.5
test: noise:0(84.53); noise:0.1(85.10); noise:0.2(86.77); noise:0.3(88.70); noise:0.4(89.85); noise:0.5(89.61);noise:0.6(84.20)

train: noise:0.6
test: noise:0(60.63); noise:0.1(62.78); noise:0.2(69.74); noise:0.3(75.32); noise:0.4(81.59); noise:0.5(86.51);noise:0.6(87.15)

train: teacher:0.1;0.5; student:0.3 epoch:40
test: noise:0(91.60); noise:0.1(91.54); noise:0.2(91.22); noise:0.3(90.57); noise:0.4(88.79); noise:0.5(83.07);noise:0.6(69.41)

train: teacher:0.1;0.5; student:0.3 epoch:200
test: noise:0(92.93); noise:0.1(93.08); noise:0.2(92.98); noise:0.3(92.40); noise:0.4(91.32); noise:0.5(83.07);noise:0.6(69.41)

bian test: noise:0(93.53); noise:0.1(93.28); noise:0.2(92.98); noise:0.3(92.40); noise:0.4(91.42); noise:0.5(88.91)
```


- **VGG8**
```
train: noise:0
test: noise:0(91.64); noise:0.1(90.58); noise:0.2(86.50); noise:0.3(78.85); noise:0.4(65.24); noise:0.5(45.38);noise:0.6(27.51)

train: noise:0.1
test: noise:0(91.22); noise:0.1(90.80); noise:0.2(88.59); noise:0.3(83.53); noise:0.4(74.76); noise:0.5(58.31);noise:0.6(36.96)

train: noise:0.2
test: noise:0(90.87); noise:0.1(90.28); noise:0.2(89.43); noise:0.3(86.51); noise:0.4(80.93); noise:0.5(67.96);noise:0.6(47.95)

train: noise:0.3
test: noise:0(89.72); noise:0.1(89.57); noise:0.2(88.99); noise:0.3(87.37); noise:0.4(83.67); noise:0.5(75.39);noise:0.6(56.02)

train: noise:0.4
test: noise:0(88.14); noise:0.1(88.22); noise:0.2(87.62); noise:0.3(86.78); noise:0.4(84.27); noise:0.5(79.45);noise:0.6(66.10)

train: noise:0.5
test: noise:0(84.08); noise:0.1(84.24); noise:0.2(85.18); noise:0.3(85.31); noise:0.4(83.91); noise:0.5(82.25);noise:0.6(74.35)

train: noise:0.6
test: noise:0(74.53); noise:0.1(75.43); noise:0.2(76.69); noise:0.3(79.41); noise:0.4(80.76); noise:0.5(80.90);noise:0.6(77.28)

test: noise:0(90.07); noise:0.1(90.00); noise:0.2(89.61); noise:0.3(88.78); noise:0.4(85.87); noise:0.5(79.99);noise:0.6()

bian test: noise:0(91.17); noise:0.1(90.20); noise:0.2(89.81); noise:0.3(88.78); noise:0.4(85.87); noise:0.5(79.99);noise:0.6()

0.1-0.5 mixed noise training
test: noise:0(90.38); noise:0.1(90.06); noise:0.2(88.83); noise:0.3(86.82); noise:0.4(82.29); noise:0.5(72.01);noise:0.6()
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
