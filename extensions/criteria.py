import torch
import torch.nn as nn
from legacy.registries import register_criterion


@register_criterion("RidgeRegressionLoss")
class RidgeRegressionLoss(nn.Module):
    """
    Example: MSE + L2 Regularization on the output.
    Usage in YAML: criterion_name: "RidgeRegressionLoss"
    """

    def __init__(self, lambda_reg: float = 0.01):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_reg = lambda_reg

    def forward(self, output, target):
        loss = self.mse(output, target)
        reg = self.lambda_reg * torch.norm(output, 2)
        return loss + reg
