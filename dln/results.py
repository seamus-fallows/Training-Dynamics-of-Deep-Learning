from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig


@dataclass
class RunResult:
    """Result from a single training run."""

    history: dict[str, list[Any]]
    config: DictConfig

    def __getitem__(self, key: str) -> list[Any]:
        return self.history[key]

    def has(self, metric: str) -> bool:
        return metric in self.history

    def final(self, metric: str) -> Any:
        return self.history[metric][-1]

    def metric_names(self) -> list[str]:
        """Return list of available metrics (excluding 'step')."""
        return [k for k in self.history.keys() if k != "step"]


@dataclass
class SweepResult:
    """Result from a parameter sweep."""

    runs: dict[str, RunResult]
    sweep_param: str
