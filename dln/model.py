import torch as t
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig


class DeepLinearNetwork(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        sizes = [cfg.in_dim] + [cfg.hidden_dim] * cfg.num_hidden + [cfg.out_dim]
        self.layers = nn.Sequential(
            *[nn.Linear(sizes[i], sizes[i + 1], bias=0) for i in range(len(sizes) - 1)]
        )

        if cfg.gamma is not None:
            # Scaling defined in https://arxiv.org/pdf/2106.15933
            std = cfg.hidden_dim ** (-cfg.gamma / 2)
            gen = t.Generator().manual_seed(cfg.model_seed)
            self._init_weights(std, gen)

    def _init_weights(self, std: float, generator: t.Generator) -> None:
        with t.no_grad():
            for layer in self.layers:
                layer.weight.data = (
                    t.randn(layer.weight.shape, generator=generator) * std
                )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
