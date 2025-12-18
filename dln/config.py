from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ModelConfig:
    in_dim: int
    hidden_dim: int
    out_dim: int
    num_hidden: int
    gamma: float
    bias: bool = False
    seed: int = 0


@dataclass
class DataConfig:
    train_samples: int
    test_samples: int | None
    # Dictionary for dataset-specific parameters e.g. mean, std.
    params: dict[str, Any] | None
    data_seed: int = 0
    noise_std: float = 0.0
    online: bool = False


@dataclass
class TrainingConfig:
    lr: float
    batch_size: int | None
    optimizer: str
    optimizer_params: dict[str, Any] | None
    criterion: str
    batch_seed: int = 0


@dataclass
class ExperimentMeta:
    name: str


@dataclass
class CallbackConfig:
    name: str
    params: dict[str, Any] | None = None


@dataclass
class MetricDataConfig:
    mode: Literal["population", "estimator"]
    holdout_size: int | None = None


@dataclass
class ExperimentConfig:
    experiment: ExperimentMeta
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    max_steps: int
    evaluate_every: int
    metrics: list[str] = field(default_factory=list)
    metric_data: MetricDataConfig | None = None
    callbacks: list[CallbackConfig] = field(default_factory=list)
    stop_threshold: float | None = None


@dataclass
class ComparativeExperimentConfig:
    experiment: ExperimentMeta
    model_a: ModelConfig
    model_b: ModelConfig
    data: DataConfig
    training_a: TrainingConfig
    training_b: TrainingConfig
    max_steps: int
    evaluate_every: int
    model_metrics: list[str] = field(default_factory=list)
    comparative_metrics: list[str] = field(default_factory=list)
    metric_data: MetricDataConfig | None = None
    callbacks_a: list[CallbackConfig] = field(default_factory=list)
    callbacks_b: list[CallbackConfig] = field(default_factory=list)
    shared: dict[str, Any] = field(default_factory=dict)
    stop_threshold: float | None = None
