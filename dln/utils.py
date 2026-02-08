import os
import yaml
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch as t
from torch import Tensor
from omegaconf import OmegaConf, DictConfig


def resolve_device(device: str) -> t.device:
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


# =============================================================================
# Save / Load
# =============================================================================


def save_history(history: dict[str, list[Any]], output_dir: Path) -> None:
    """Save training history as compressed numpy archive (atomic write)."""
    history_path = output_dir / "history.npz"
    tmp_path = output_dir / "history.tmp.npz"
    np.savez(tmp_path, **{k: np.array(v) for k, v in history.items()})
    os.replace(tmp_path, history_path)


def save_overrides(overrides: dict, output_dir: Path) -> None:
    with (output_dir / "overrides.json").open("w") as f:
        json.dump(overrides, f)


def config_diff(existing: dict, new: dict, prefix: str = "") -> list[str]:
    """Return human-readable differences between two config dicts."""
    diffs = []
    all_keys = sorted(existing.keys() | new.keys())
    for key in all_keys:
        full_key = f"{prefix}.{key}" if prefix else key
        if key not in existing:
            diffs.append(f"  + {full_key} = {new[key]!r}")
        elif key not in new:
            diffs.append(f"  - {full_key} = {existing[key]!r}")
        elif isinstance(existing[key], dict) and isinstance(new[key], dict):
            diffs.extend(config_diff(existing[key], new[key], full_key))
        elif existing[key] != new[key]:
            diffs.append(f"  {full_key}: {existing[key]!r} -> {new[key]!r}")
    return diffs


def save_sweep_config(config: dict, output_dir: Path) -> None:
    """Save resolved config at sweep root, verifying consistency if already exists."""
    path = output_dir / "config.yaml"
    if path.exists():
        with path.open("r") as f:
            existing = yaml.safe_load(f)
        if existing != config:
            diffs = config_diff(existing, config)
            diff_str = "\n".join(diffs)
            raise ValueError(
                f"Config mismatch in {output_dir}:\n{diff_str}\n\n"
                f"If resuming, re-run with the original parameters.\n"
                f"If this is a new experiment, use a different --output."
            )
        return
    with path.open("w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def load_history(output_dir: Path) -> dict[str, np.ndarray]:
    with np.load(output_dir / "history.npz") as data:
        return {k: data[k] for k in data.files}


def load_run(path: Path) -> dict:
    """Load a single run's history, config, and overrides."""
    history = load_history(path)

    config = None
    for config_dir in [path, path.parent]:
        config_path = config_dir / "config.yaml"
        if config_path.exists():
            with config_path.open("r") as f:
                config = yaml.safe_load(f)
            break

    overrides = {}
    overrides_path = path / "overrides.json"
    if overrides_path.exists():
        with overrides_path.open("r") as f:
            overrides = json.load(f)

    return {"history": history, "config": config, "overrides": overrides}


def load_sweep(sweep_dir: Path) -> dict:
    """Load all results from a sweep directory.

    Returns dict with 'config' and 'runs' keys.
    Each run has 'history', 'overrides', and 'subdir' keys.
    """
    config = None
    config_path = sweep_dir / "config.yaml"
    if config_path.exists():
        with config_path.open("r") as f:
            config = yaml.safe_load(f)

    runs = []
    for history_path in sorted(sweep_dir.rglob("history.npz")):
        job_dir = history_path.parent
        if job_dir == sweep_dir:
            subdir = ""
        else:
            subdir = str(job_dir.relative_to(sweep_dir))

        entry = {
            "subdir": subdir,
            "history": load_history(job_dir),
        }
        overrides_path = job_dir / "overrides.json"
        if overrides_path.exists():
            with overrides_path.open("r") as f:
                entry["overrides"] = json.load(f)
        else:
            entry["overrides"] = {}
        runs.append(entry)

    return {"config": config, "runs": runs}


# =============================================================================
# Config Resolution
# =============================================================================

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
    """Apply overrides, merge shared configs and resolve."""
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
