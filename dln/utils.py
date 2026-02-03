import json
import os
import random
from pathlib import Path
from typing import Any, Type
from torch import nn
import numpy as np
import torch as t
from torch.optim import Optimizer
from torch import Tensor
from omegaconf import OmegaConf, DictConfig


def seed_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)


def resolve_device(device: str) -> t.device:
    """Convert device string to torch.device."""
    if device == "cpu":
        return t.device("cpu")
    if device == "cuda":
        n_gpus = t.cuda.device_count()
        if n_gpus > 1:
            gpu_id = os.getpid() % n_gpus
            return t.device(f"cuda:{gpu_id}")
        return t.device("cuda")
    if device == "mps":
        return t.device("mps")
    raise ValueError(f"Unknown device: {device}")


def to_device(
    data: tuple[Tensor, Tensor] | None, device: t.device
) -> tuple[Tensor, Tensor] | None:
    if data is None:
        return None
    inputs, targets = data
    return inputs.to(device), targets.to(device)


def get_optimizer_cls(name: str) -> Type[Optimizer]:
    """Resolve optimizer class from torch.optim by name."""
    try:
        return getattr(t.optim, name)
    except AttributeError as e:
        raise ValueError(f"Unknown optimizer: '{name}'") from e


def get_criterion_cls(name: str) -> Type[nn.Module]:
    """Resolve loss criterion class from torch.nn by name."""
    try:
        return getattr(nn, name)
    except AttributeError as e:
        raise ValueError(f"Unknown criterion: '{name}'") from e


def save_history(history: dict[str, list[Any]], output_dir: Path) -> None:
    """Save training history to JSON file (columnar format)."""
    history_path = output_dir / "history.json"
    with history_path.open("w") as f:
        json.dump(history, f)


def load_history(output_dir: Path) -> dict[str, list[Any]]:
    """Load training history from JSON file (columnar format)."""
    history_path = output_dir / "history.json"
    with history_path.open("r") as f:
        return json.load(f)


def rows_to_columns(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Convert row-oriented history to columnar format."""
    columns: dict[str, list[Any]] = {key: [] for key in rows[0].keys()}
    for record in rows:
        for key, value in record.items():
            columns[key].append(value)
    return columns


def validate_config(cfg: DictConfig, config_type: str = "single") -> None:
    """Validate config values before experiment execution."""

    if config_type == "single":
        _validate_model_config(cfg.model)
        _validate_training_config(cfg.training)
        _validate_data_config(cfg.data)

    elif config_type == "comparative":
        _validate_model_config(cfg.model_a)
        _validate_model_config(cfg.model_b)
        _validate_training_config(cfg.training_a)
        _validate_training_config(cfg.training_b)
        _validate_data_config(cfg.data)


def _validate_model_config(model_cfg: DictConfig) -> None:
    assert model_cfg.in_dim > 0, "in_dim must be positive"
    assert model_cfg.out_dim > 0, "out_dim must be positive"
    assert model_cfg.num_hidden >= 0, "num_hidden must be non-negative"
    assert model_cfg.hidden_dim > 0, "hidden_dim must be positive"
    if model_cfg.gamma is not None:
        assert model_cfg.gamma > 0, "gamma must be positive"


def _validate_training_config(training_cfg: DictConfig) -> None:
    assert training_cfg.lr > 0, "lr must be positive"
    if training_cfg.batch_size is not None:
        assert training_cfg.batch_size > 0, "batch_size must be positive"


def _validate_data_config(data_cfg: DictConfig) -> None:
    assert data_cfg.train_samples > 0, "train_samples must be positive"
    if data_cfg.test_samples is not None:
        assert data_cfg.test_samples > 0, "test_samples must be positive"
    assert data_cfg.noise_std >= 0, "noise_std must be non-negative"

    if data_cfg.online and data_cfg.test_samples is None:
        raise ValueError("Online mode requires test_samples for loss tracking")


CONFIG_ROOT = Path(__file__).parent.parent / "configs"


def load_config(
    config_name: str,
    config_dir: str = "single",
    overrides: dict[str, Any] | None = None,
) -> DictConfig:
    """Load a YAML config and apply overrides."""
    config_path = CONFIG_ROOT / config_dir / f"{config_name}.yaml"
    cfg = OmegaConf.load(config_path)

    if overrides:
        for key, value in overrides.items():
            OmegaConf.update(cfg, key, value, merge=True)

    # Merge shared into a/b configs for comparative experiments
    if config_dir == "comparative" and "shared" in cfg:
        if "model" in cfg.shared:
            cfg.model_a = OmegaConf.merge(cfg.shared.model, cfg.get("model_a") or {})
            cfg.model_b = OmegaConf.merge(cfg.shared.model, cfg.get("model_b") or {})
        if "training" in cfg.shared:
            cfg.training_a = OmegaConf.merge(
                cfg.shared.training, cfg.get("training_a") or {}
            )
            cfg.training_b = OmegaConf.merge(
                cfg.shared.training, cfg.get("training_b") or {}
            )

    OmegaConf.resolve(cfg)

    validate_config(cfg, config_dir)

    return cfg
