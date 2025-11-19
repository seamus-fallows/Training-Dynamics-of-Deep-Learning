import torch as t
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer


class SGF(Optimizer):
    def __init__(self, params, lr=1e-3, mode="full"):
        defaults = dict(lr=lr)
        super(SGF, self).__init__(params, defaults)

    @t.no_grad()
    def step(self, closure=None):
        pass
