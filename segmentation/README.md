# MTRD-MNI-CIM for segmentation

## Project Structure
- `dataset.py`: Defines the `CarvanaDataset` class for loading and preprocessing the image data.
- `model.py`: Implements the U-Net architecture using PyTorch.
- `train.py`: Contains the main training loop and configuration settings.
- `train_nonid.py`: Contains the main training loop and configuration settings for non-identity mapping.
- `train_nonid_pcm.py`: Contains the main training loop and configuration settings for non-identity mapping with PCM.
- `utils.py`: Provides utility functions for data loading, model checkpointing, and evaluation.


## Requirements

- Python 3.x
- PyTorch
- torchvision
- PIL
- numpy
- albumentations
- tqdm


## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ShutterSpeedLabs/UNetFromScratch.git
   cd UNetFromScratch
2. Install the required packages:
   pip install torch torchvision pillow numpy albumentations tqdm

3. Download the Carvana Image Masking Challenge dataset and update the path variables in train.py:
   TRAIN_IMG_DIR = "path/to/train/images"
   TRAIN_MASK_DIR = "path/to/train/masks"
   VAL_IMG_DIR = "path/to/val/images"
   VAL_MASK_DIR = "path/to/val/masks"

## Usage
To train the model:
python train.py

The script will automatically save checkpoints, evaluate accuracy, and save prediction images during training.
## Model Architecture
The U-Net architecture is implemented in model.py. It consists of:

Encoder (downsampling path)
Bottleneck
Decoder (upsampling path)
Skip connections

## Data Augmentation
The project uses the albumentations library for data augmentation. The augmentations include:

Resizing
Rotation
Horizontal and vertical flips
Normalization

## Evaluation
The model's performance is evaluated using:

Pixel-wise accuracy
Dice score

## Customization
You can modify the hyperparameters in train.py to experiment with different settings:
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 3
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
## Acknowledgements
This project is based on the U-Net architecture and is designed for the Carvana Image Masking Challenge.

