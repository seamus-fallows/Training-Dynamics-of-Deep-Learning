from dataclasses import dataclass
from typing import Literal, Dict, Any
from hydra.core.config_store import ConfigStore


@dataclass
class ModelConfig:
    in_dim: int
    out_dim: int
    num_hidden: int
    hidden_size: int
    gamma: float
    bias: bool = False


@dataclass
class DataConfig:
    type: Literal["diagonal_teacher", "random_teacher"]

    num_samples: int
    test_split: float | None
    data_seed: int = 0

    # Dictionary for dataset-specific parameters.
    params: Dict[str, Any] | None = None


@dataclass
class TrainingConfig:
    lr: float
    max_steps: int
    batch_size: int | None  # None means full batch
    evaluate_every: int

    optimizer: str = "SGD"
    criterion: str = "MSELoss"
    model_seed: int = 0


@dataclass
class ExperimentMeta:
    name: str


@dataclass
class ExperimentConfig:
    experiment: ExperimentMeta
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig


def register_configs():
    cs = ConfigStore.instance()
    # This registers the schema "ExperimentConfig" as the default base configuration
    cs.store(name="base_config", node=ExperimentConfig)
