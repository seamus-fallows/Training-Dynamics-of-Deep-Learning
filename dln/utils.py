import json
import random
from pathlib import Path
from typing import Any, Type
from torch import nn
import numpy as np
import torch as t
from torch.optim import Optimizer
from torch import Tensor


def seed_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)


def get_device() -> t.device:
    if t.cuda.is_available():
        return t.device("cuda")
    elif t.backends.mps.is_available():
        return t.device("mps")
    else:
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
