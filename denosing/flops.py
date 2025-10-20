import torch
from torchsummary import summary
from thop import profile
from thop import clever_format
from models import DnCNN,DnCNN_smi
from models import *
# Assume we have a pre-trained model
model = DnCNN(channels=1)
model.eval()

# Use thop to analyze model FLOPs and parameters
input = torch.randn(1, 1, 256, 256)  # Randomly generate an input tensor, this size should match the model input size
flops, params = profile(model, inputs=(input,))

# Convert results to a more readable format
flops, params = clever_format([flops, params], '%.3f')

print(f"FLOPs: {flops}, Parameters: {params}")

