"""Plotting utilities for training experiments."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="colorblind")

History = list[dict[str, Any]]


def _smooth(values: list[float], window: int) -> list[float]:
    """Trailing moving average."""
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed.append(sum(values[start : i + 1]) / (i - start + 1))
    return smoothed


def _get_steps(history: History) -> list[int]:
    return [h["step"] for h in history]


def _get_values(history: History, metric: str) -> list[float]:
    return [h[metric] for h in history]


def _save_and_show(fig: plt.Figure, save_path: Path | str | None, show: bool) -> None:
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def infer_metrics(history: History) -> list[str]:
    """Auto-detect plottable metrics from history."""
    if not history:
        return []

    keys = set(history[0].keys()) - {"step"}

    loss_metrics = sorted(k for k in keys if "loss" in k)
    comparative = sorted(k for k in keys if k in {"param_distance", "param_cosine_sim"})

    other = keys - set(loss_metrics) - set(comparative)
    model_metrics = sorted(k for k in other if not k.endswith(("_a", "_b")))

    return loss_metrics + model_metrics + comparative


def is_comparative(history: History) -> bool:
    if not history:
        return False
    return "train_loss_a" in history[0]


def plot_single(
    history: History,
    metrics: list[str] | None = None,
    save_dir: Path | str | None = None,
    show: bool = True,
    log_scale: bool = True,
    smoothing: int | None = None,
) -> list[plt.Figure]:
    """Plot metrics from a single training run. One figure per metric."""
    if metrics is None:
        metrics = infer_metrics(history)

    if not metrics:
        raise ValueError("No metrics to plot")

    steps = _get_steps(history)
    save_dir = Path(save_dir) if save_dir else None
    figures = []

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))

        values = _get_values(history, metric)
        if smoothing:
            values = _smooth(values, smoothing)
        ax.plot(steps, values, linewidth=1.5)

        if log_scale and "loss" in metric:
            ax.set_yscale("log")

        ax.set_xlabel("Step")
        ax.set_ylabel(metric)
        ax.set_title(metric.replace("_", " ").title())
        fig.tight_layout()

        save_path = save_dir / f"{metric}.png" if save_dir else None
        _save_and_show(fig, save_path, show)
        figures.append(fig)

    return figures


def plot_comparative(
    history: History,
    metrics: list[str] | None = None,
    save_dir: Path | str | None = None,
    show: bool = True,
    log_scale: bool = True,
    smoothing: int | None = None,
) -> list[plt.Figure]:
    """Plot metrics from a comparative run. One figure per metric, with _a/_b paired."""
    if metrics is None:
        metrics = infer_metrics(history)

    if not metrics:
        raise ValueError("No metrics to plot")

    steps = _get_steps(history)

    # Group into paired (_a/_b) and standalone
    paired: list[str] = []
    standalone: list[str] = []

    seen_bases = set()
    for m in metrics:
        if m.endswith("_a") or m.endswith("_b"):
            base = m[:-2]
            if base not in seen_bases:
                paired.append(base)
                seen_bases.add(base)
        else:
            standalone.append(m)

    figures = []
    save_dir = Path(save_dir) if save_dir else None

    for base in paired:
        fig, ax = plt.subplots(figsize=(10, 6))

        for suffix, label in [("_a", "Model A"), ("_b", "Model B")]:
            key = f"{base}{suffix}"
            if key in history[0]:
                values = _get_values(history, key)
                if smoothing:
                    values = _smooth(values, smoothing)
                ax.plot(steps, values, label=label, linewidth=1.5)

        if log_scale and "loss" in base:
            ax.set_yscale("log")

        ax.set_xlabel("Step")
        ax.set_ylabel(base)
        ax.set_title(base.replace("_", " ").title())
        ax.legend()
        fig.tight_layout()

        save_path = save_dir / f"{base}.png" if save_dir else None
        _save_and_show(fig, save_path, show)
        figures.append(fig)

    for metric in standalone:
        fig, ax = plt.subplots(figsize=(10, 6))

        values = _get_values(history, metric)
        if smoothing:
            values = _smooth(values, smoothing)
        ax.plot(steps, values, linewidth=1.5)

        if log_scale and "loss" in metric:
            ax.set_yscale("log")

        ax.set_xlabel("Step")
        ax.set_ylabel(metric)
        ax.set_title(metric.replace("_", " ").title())
        fig.tight_layout()

        save_path = save_dir / f"{metric}.png" if save_dir else None
        _save_and_show(fig, save_path, show)
        figures.append(fig)

    return figures


def plot_sweep(
    histories: dict[str, History],
    metric: str = "train_loss",
    save_path: Path | str | None = None,
    show: bool = True,
    log_scale: bool = True,
    smoothing: int | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Plot a single metric across multiple runs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, history in histories.items():
        steps = _get_steps(history)
        values = _get_values(history, metric)
        if smoothing:
            values = _smooth(values, smoothing)
        ax.plot(steps, values, label=label, linewidth=1.5)

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Step")
    ax.set_ylabel(metric)
    ax.legend()

    if title:
        ax.set_title(title)

    fig.tight_layout()
    _save_and_show(fig, save_path, show)

    return fig
