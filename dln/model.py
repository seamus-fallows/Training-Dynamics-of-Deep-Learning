import torch as t
import torch.nn as nn
from torch import Tensor
from .config import ModelConfig


class DeepLinearNetwork(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()

        self.config = config

        sizes = (
            [config.in_dim] + [config.hidden_dim] * config.num_hidden + [config.out_dim]
        )
        self.model = nn.Sequential(
            *[
                nn.Linear(sizes[i], sizes[i + 1], bias=config.bias)
                for i in range(len(sizes) - 1)
            ]
        )
        if config.gamma is not None:
            std = config.hidden_dim ** (-config.gamma / 2)
            self._init_weights(std)

    def _init_weights(self, std: float) -> None:
        with t.no_grad():
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=std)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
