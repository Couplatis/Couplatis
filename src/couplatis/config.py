import torch

from pathlib import Path


BASE_DIR = Path(".").resolve()


class Config:
    """Config"""
    epoch = 1
    batchsize = 256
    num_workers = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __repr__(self) -> str:
        """repr"""
        return f"<Config epoch={self.epoch}, batch_size={self.batchsize},
          num_workers={self.num_workers}, device={self.device!r}>"
