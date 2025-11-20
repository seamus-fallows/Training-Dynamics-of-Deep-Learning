from dataclasses import dataclass
import torch.optim as optim
import torch.nn as nn
from typing import List, Any, Dict


@dataclass
class DeepLinearNetworkConfig:
    num_hidden: int
    hidden_size: int
    in_size: int
    out_size: int
    gamma: float | None
    bias: bool = False


@dataclass
class TrainingConfig:
    lr: float
    max_steps: int
    evaluate_every: int
    optimizer_cls: type[optim.Optimizer] = optim.SGD
    criterion_cls: type[nn.Module] = nn.MSELoss
    batch_size: int | None = None  # None means full batch training


@dataclass
class DataConfig:
    name: str = ""
    num_samples: int
    in_size: int
    test_split: float | None
    seed: int


@dataclass
class DiagonalTeacherConfig(DataConfig):
    name: str = "diagonal_teacher"
    scale_factor: float


@dataclass
class RandomTeacherConfig(DataConfig):
    name: str = "random_teacher"
    scale_factor: float
    mean: float
    std: float


@dataclass
class ExperimentConfig:
    name: str
    dln_config: DeepLinearNetworkConfig
    training_config: TrainingConfig
    data_config: DataConfig
    # seed for model initialization and dataloader shuffling
    model_seed: int


@dataclass
class GridSearchConfig:
    """
    Defines a sweep.
    param_grid keys should use dot notation (e.g. 'training_config.lr')
    """

    name: str
    base_config: ExperimentConfig
    param_grid: Dict[str, List[Any]]
