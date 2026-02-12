import os
import yaml
from pathlib import Path
from typing import Any
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
# Config Save / Diff
# =============================================================================


def config_diff(existing: dict, new: dict, prefix: str = "") -> list[str]:
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
    """Raises ValueError if an existing config.yaml differs from the new one."""
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


# =============================================================================
# Config Resolution
# =============================================================================

CONFIG_ROOT = Path(__file__).parent.parent / "configs"


def load_base_config(config_name: str, config_dir: str = "single") -> dict:
    """Returns a plain dict — no overrides applied, no interpolation resolved."""
    config_path = CONFIG_ROOT / config_dir / f"{config_name}.yaml"
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg)


def resolve_config(
    base_config: dict,
    config_dir: str = "single",
    overrides: dict[str, Any] | None = None,
) -> DictConfig:
    """Apply overrides, merge shared→model_a/b for comparative configs, and resolve."""
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
