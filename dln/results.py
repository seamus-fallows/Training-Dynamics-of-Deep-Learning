from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig


@dataclass
class RunResult:
    """Result from a single training run."""

    history: dict[str, list[Any]]
    config: DictConfig
    output_dir: Path | None = None

    def __getitem__(self, key: str) -> list[Any]:
        return self.history[key]

    def has(self, metric: str) -> bool:
        return metric in self.history

    def final(self, metric: str) -> Any:
        return self.history[metric][-1]

    def metrics(self) -> list[str]:
        """Return list of available metrics (excluding 'step')."""
        return [k for k in self.history.keys() if k != "step"]


@dataclass
class SweepResult:
    """Result from a parameter sweep."""

    runs: dict[str, list[RunResult]]
    sweep_param: str

    def param_values(self) -> list[str]:
        return list(self.runs.keys())

    def flatten(self) -> dict[str, RunResult]:
        """Flatten to individual runs for plotting."""
        return {
            f"{param}_{i}": run
            for param, runs in self.runs.items()
            for i, run in enumerate(runs)
        }
