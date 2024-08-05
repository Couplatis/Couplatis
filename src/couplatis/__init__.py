import torch

torch.cuda.set_device(0)
print(torch.cuda.is_available())
print(torch.__version__)
