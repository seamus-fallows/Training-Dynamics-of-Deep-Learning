from dataclasses import dataclass
import torch.optim as optim
import torch.nn as nn


@dataclass
class DeepLinearNetworkConfig:
    num_hidden: int
    hidden_size: int
    in_size: int
    out_size: int
    bias: bool = False


@dataclass
class TrainingConfig:
    num_epochs: int
    lr: float
    optimizer_cls: type[optim.Optimizer] = optim.SGD
    criterion_cls: type[nn.Module] = nn.MSELoss
    evaluate_every: int = 50  # Evaluate test loss every k epochs
    batch_size: int | None = None  # None means full batch training


@dataclass
class TeacherStudentExperimentConfig:
    name: str
    dln_config: DeepLinearNetworkConfig
    training_config: TrainingConfig
    gamma: float  # Weights are initialised with variance = hidden_size^(-gamma)
    num_samples: int = 125
    teacher_matrix_scale_factor: float = 10.0
    test_split: float | None = 0.2
    base_seed: int = 69
