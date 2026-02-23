import torch as t
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig


class DeepLinearNetwork(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        sizes = [cfg.in_dim] + [cfg.hidden_dim] * cfg.num_hidden + [cfg.out_dim]
        self.layers = nn.Sequential(
            *[
                nn.Linear(d_in, d_out, bias=False)
                for d_in, d_out in zip(sizes, sizes[1:])
            ]
        )

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

    def end_to_end_weight(self) -> Tensor:
        return t.linalg.multi_dot([layer.weight for layer in reversed(self.layers)])

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
