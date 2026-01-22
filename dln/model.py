import torch as t
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig


class DeepLinearNetwork(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()

        sizes = [cfg.in_dim] + [cfg.hidden_dim] * cfg.num_hidden + [cfg.out_dim]
        self.model = nn.Sequential(
            *[
                nn.Linear(sizes[i], sizes[i + 1], bias=cfg.bias)
                for i in range(len(sizes) - 1)
            ]
        )
        if cfg.gamma is not None:
            # Scaling defined in https://arxiv.org/pdf/2106.15933
            std = cfg.hidden_dim ** (-cfg.gamma / 2)
            self._init_weights(std)

    def _init_weights(self, std: float) -> None:
        with t.no_grad():
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=std)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
