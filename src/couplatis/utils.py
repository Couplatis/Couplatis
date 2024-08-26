"""
utils
"""
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from couplatis.config import Config


def get_train_loader(config: Config) -> DataLoader[datasets.CIFAR10]:
    """Get loader for training"""
    return DataLoader(
        datasets.CIFAR10(
            "data\\models", train=True, transform=transforms.ToTensor(), download=True
        ),
        batch_size=config.batchsize,
        shuffle=True,
        num_workers=config.num_workers,
    )


def get_test_loader(config: Config) -> DataLoader[datasets.CIFAR10]:
    """Get loader for testing"""
    return DataLoader(
        datasets.CIFAR10(
            "data\\models", train=False, transform=transforms.ToTensor(), download=True
        ),
        batch_size=config.batchsize,
        shuffle=False,
        num_workers=config.num_workers,
    )
