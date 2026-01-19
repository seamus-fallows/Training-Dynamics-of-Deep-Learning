import json
import random
from pathlib import Path
from typing import Any, Type
from torch import nn
import numpy as np
import torch as t
from torch.optim import Optimizer
from torch import Tensor
import os
from omegaconf import OmegaConf, DictConfig


def seed_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)


def get_device() -> t.device:
    if t.cuda.is_available():
        n_gpus = t.cuda.device_count()
        if n_gpus > 1:
            # Assign by PID so each worker keeps the same GPU
            gpu_id = os.getpid() % n_gpus
            return t.device(f"cuda:{gpu_id}")
        return t.device("cuda")
    if t.backends.mps.is_available():
        return t.device("mps")
    return t.device("cpu")


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
        json.dump(history, f, indent=2)


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

    OmegaConf.resolve(cfg)

    # Compute derived values
    if "num_evaluations" in cfg and "evaluate_every" not in cfg:
        OmegaConf.update(
            cfg, "evaluate_every", max(1, cfg.max_steps // cfg.num_evaluations)
        )

    return cfg
