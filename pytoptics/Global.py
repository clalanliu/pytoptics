import torch

torchdevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torchPrecision = torch.DoubleTensor

def set_torch_device(device):
    global torchdevice
    torchdevice = device

def set_torch_precision(precision):
    global torchPrecision
    torchPrecision = precision