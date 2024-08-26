"""
model
"""

import torch

from pathlib import Path
from typing import Union
from torch.nn import functional as F

from couplatis.config import Config



class EpsNet:
    """ EpsNet"""
    def __init__(
        self,
        x_size: int,
        y_size: int,
        a_size: int,
        alpha: float,
        beta: float,
        js_mall: float,
        eps: float,
        config: Config,
    ):
        self.x_size = x_size
        self.y_size = y_size
        self.a_size = a_size
        self.w1 = torch.zeros(y_size, x_size, device=config.device)  
        self.w2 = torch.zeros(a_size, y_size, device=config.device)  
        self.b = torch.zeros(y_size, device=config.device)  
        self.alpha = alpha
        self.beta = beta
        self.js_mall = js_mall
        self.eps = eps
        self.config = config
        torch.nn.init.uniform_(self.w1, -0.1, 0.1)
        torch.nn.init.uniform_(self.w2, -0.1, 0.1)

    def neuronDTE(self, gamma, p):
        """neuronDTE"""
        return (1 - self.eps) * p + self.eps * torch.pow(
            torch.sin(p + torch.cos(p + gamma)), 2
        )

    def neuronADE(self, gamma, p, q):
        """neuronADE"""
        p = (1 - self.eps) * p + self.eps * torch.pow(
            torch.sin(p + torch.cos(p + gamma)), 2
        )
        q = (1 - self.eps) * q + self.eps * torch.sin(
            2 * (p + torch.cos(p + gamma)) * (q - torch.sin(p + gamma) * (q + 1))
        )
        return [p, q]

    def cost(self, pred, label):
        """cost"""
        J = -pred[range(label.shape[0]), label].log().mean()
        return J

    @torch.no_grad()
    def train(self, batch):
        """train"""
        batch_size = batch[1].shape[0]
        y = torch.zeros(batch_size, self.y_size, device=self.config.device)   # b, ysize
        p = torch.zeros(batch_size, self.y_size, device=self.config.device)   # b, ysize
        q = torch.zeros(batch_size, self.y_size, device=self.config.device)   # b, ysize
        counter = 0
        while True:
            # init
            cnt = 0.0
            batch[0] = batch[0].to(self.config.device)  
            batch[1] = batch[1].to(self.config.device)  

            x = batch[0].view(batch_size, -1)  # b, xsize
            d = batch[1]  # b, 1

            # forward
            # (b, xsize) x (xsize, ysize) + (ysize,) -> (b, ysize)
            gamma = torch.matmul(x, self.w1.T) + self.b
            y = self.neuronDTE(gamma, y)
            z = torch.matmul(y, self.w2.T)
            a = F.softmax(z, dim=1)  # b, a_size

            cnt += (a.argmax(dim=1) == d).sum()
            J = self.cost(a, d)

            delta_z = a - F.one_hot(d, num_classes=self.a_size)  # b, a_size
            delta_z /= batch_size

            delta_v = (
                -self.eps
                * torch.sin(2 * (p + torch.cos(p + gamma)))
                * (torch.matmul(delta_z, self.w2))
            )

            # update
            # (a_size, b) x (b, ysize) -> (a_size, ysize)
            self.w2 -= self.beta * torch.matmul(delta_z.T, y)
            # (ysize, b) x (b, xsize) -> (ysize, xsize)
            self.w1 -= self.alpha * torch.matmul(
                (delta_v * (1 + q)).T, x
            )  # torch.matmul(eta0.T, x)
            # (b, ysize) -> (ysize,)
            self.b -= self.alpha * (delta_v * (1 + q)).sum(dim=0)

            p, q = self.neuronADE(gamma, p, q)

            batch_acc = cnt / batch_size
            counter += 1
            if J < self.js_mall or counter > 100:
                break

        return batch_acc

    @torch.no_grad()
    def test(self, batch):
        """test"""
        # init
        batch[0] = batch[0].to(self.config.device)  
        batch_size = batch[1].shape[0]
        y = torch.zeros(batch_size, self.y_size, device=self.config.device)   # b, ysize

        cnt = 0
        while True:
            x = batch[0].view(batch_size, -1)  # b, xsize

            gamma = torch.matmul(x, self.w1.T) + self.b
            y_ = self.neuronDTE(gamma, y)
            z = torch.matmul(y_, self.w2.T)
            a = F.softmax(z, dim=1)  # b, a_size
            cnt += 1
            if cnt >= 20:
                break

        return a.detach().cpu().numpy()

    def save(self, path: Union[Path, str]):
        """save"""
        torch.save({"w1": self.w1, "b": self.b, "w2": self.w2}, path)

    def load(self, path):
        """load"""
        params = torch.load(path, map_location=self.config.device)  
        self.w1 = params["w1"]
        self.w2 = params["w2"]
        self.b = params["b"]
