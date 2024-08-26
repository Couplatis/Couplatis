import torch
import os

import numpy as np
import torch.utils
import torch.utils.data

from torch.nn import functional as F
from torchvision import transforms, datasets
from loguru import logger
from tqdm import tqdm

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore

class Config:
    epoch = 1
    batchsize = 256
    num_workers = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_train_loader( ):
    
    config = Config() # type: ignore

    torch.manual_seed(42)
    train_transforms = transforms.Compose(
        [
            transforms.Pad(2),
            transforms.RandomCrop(28),
            transforms.RandomAffine(degrees=15),
            transforms.ToTensor(),
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "cifar10", train=True, transform=transforms.ToTensor(), download=True
        ),
        batch_size=config.batchsize,
        shuffle=True,
        num_workers=config.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "cifar10", train=False, transform=transforms.ToTensor(), download=True
        ),
        batch_size=config.batchsize,
        shuffle=False,
        num_workers=config.num_workers,
    )
