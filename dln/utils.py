import os
import yaml
import json
from pathlib import Path
from typing import Any, Type
from torch import nn
import numpy as np
import torch as t
from torch.optim import Optimizer
from torch import Tensor
from omegaconf import OmegaConf, DictConfig


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
    """Save training history as compressed numpy archive."""
    history_path = output_dir / "history.npz"
    np.savez_compressed(history_path, **{k: np.array(v) for k, v in history.items()})


def load_history(output_dir: Path) -> dict[str, np.ndarray]:
    """Load training history from numpy archive."""
    with np.load(output_dir / "history.npz") as data:
        return {k: data[k] for k in data.files}


def save_config(cfg: dict, path: Path) -> None:
    """Save config as YAML."""

    with path.open("w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)


def save_overrides(overrides: dict, path: Path) -> None:
    """Save per-job overrides as JSON."""
    with path.open("w") as f:
        json.dump(overrides, f)


def save_base_config(config: dict, output_dir: Path) -> None:
    """Save base config at sweep root, verifying consistency if already exists."""
    path = output_dir / "config.yaml"
    if path.exists():
        with path.open("r") as f:
            existing = yaml.safe_load(f)
        if existing != config:
            raise ValueError(
                f"Base config mismatch in {output_dir}. "
                f"Cannot write results from different base configs to the same directory."
            )
        return
    save_config(config, path)


def load_run(path: Path) -> dict:
    """Load a single run's history and config."""
    history = load_history(path)
    config_path = path / "config.yaml"
    config = None
    if config_path.exists():
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
    return {"history": history, "config": config}


def load_sweep(sweep_dir: Path) -> list[dict]:
    """Load all results from a sweep directory.

    Returns list of dicts with 'history', 'overrides', and 'subdir' keys.
    """
    results = []
    for history_path in sorted(sweep_dir.rglob("history.npz")):
        job_dir = history_path.parent
        entry = {
            "subdir": str(job_dir.relative_to(sweep_dir)),
            "history": load_history(job_dir),
        }
        overrides_path = job_dir / "overrides.json"
        if overrides_path.exists():
            with overrides_path.open("r") as f:
                entry["overrides"] = json.load(f)
        else:
            entry["overrides"] = {}
        results.append(entry)
    return results


def rows_to_columns(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Convert row-oriented history to columnar format."""
    columns: dict[str, list[Any]] = {key: [] for key in rows[0].keys()}
    for record in rows:
        for key, value in record.items():
            columns[key].append(value)
    return columns


CONFIG_ROOT = Path(__file__).parent.parent / "configs"


def load_base_config(config_name: str, config_dir: str = "single") -> dict:
    """Load a YAML config file and return as a plain dict (no overrides, no resolution)."""
    config_path = CONFIG_ROOT / config_dir / f"{config_name}.yaml"
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg)


def resolve_config(
    base_config: dict,
    config_dir: str = "single",
    overrides: dict[str, Any] | None = None,
) -> DictConfig:
    """Apply overrides, merge shared configs and resolve"""
    cfg = OmegaConf.create(base_config)

    if overrides:
        for key, value in overrides.items():
            OmegaConf.update(cfg, key, value, merge=True)

    if config_dir == "comparative" and "shared" in cfg:
        if "model" in cfg.shared:
            cfg.model_a = OmegaConf.merge(cfg.shared.model, cfg.model_a)
            cfg.model_b = OmegaConf.merge(cfg.shared.model, cfg.model_b)
        if "training" in cfg.shared:
            cfg.training_a = OmegaConf.merge(cfg.shared.training, cfg.training_a)
            cfg.training_b = OmegaConf.merge(cfg.shared.training, cfg.training_b)

    OmegaConf.resolve(cfg)
    return cfg


def load_config(
    config_name: str,
    config_dir: str = "single",
    overrides: dict[str, Any] | None = None,
) -> DictConfig:
    """Load a YAML config and apply overrides."""
    base = load_base_config(config_name, config_dir)
    return resolve_config(base, config_dir, overrides)
