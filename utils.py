import torch as t
import torch.nn as nn


def initialize_dln_weights(model: nn.Module, std: float) -> None:
    """Initializes Linear layer weights with a specified standard deviation."""
    with t.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
