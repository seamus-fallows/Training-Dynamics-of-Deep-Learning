import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy import stats

from .results import RunResult


# =============================================================================
# Core helpers
# =============================================================================


def compute_ci(
    curves: list[np.ndarray] | list[list[float]], ci: float = 0.95
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (mean, lower, upper)."""
    curves = np.asarray(curves)
    n = len(curves)
    mean = curves.mean(axis=0)
    sem = curves.std(axis=0, ddof=1) / np.sqrt(n)
    t_val = stats.t.ppf((1 + ci) / 2, df=n - 1)
    lower = mean - t_val * sem
    upper = mean + t_val * sem
    return mean, lower, upper


def smooth(values: list[float] | np.ndarray, window: int) -> np.ndarray:
    """Moving average with edge padding to preserve length."""
    values = np.asarray(values)
    kernel = np.ones(window) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


# =============================================================================
# Internal helpers
# =============================================================================


def _plot_series(
    ax: Axes,
    series: RunResult | list[RunResult],
    metric: str,
    label: str | None,
    ci: float,
    smoothing: int | None,
) -> None:
    """Plot a single RunResult or list of RunResults (with CI)."""
    if isinstance(series, RunResult):
        series = [series]

    if len(series) == 1:
        steps = np.asarray(series[0]["step"])
        values = np.asarray(series[0][metric])
        if smoothing:
            values = smooth(values, smoothing)
        ax.plot(steps, values, label=label)
    else:
        steps = np.asarray(series[0]["step"])
        curves = [np.asarray(r[metric]) for r in series]
        mean, lower, upper = compute_ci(curves, ci)
        if smoothing:
            mean = smooth(mean, smoothing)
            lower = smooth(lower, smoothing)
            upper = smooth(upper, smoothing)
        (line,) = ax.plot(steps, mean, label=label)
        ax.fill_between(steps, lower, upper, alpha=0.2, color=line.get_color())


# =============================================================================
# Main plotting function
# =============================================================================


def plot(
    data: RunResult
    | list[RunResult]
    | dict[str, RunResult | list[RunResult]],
    metric: str = "test_loss",
    ylabel: str | None = None,
    ci: float = 0.95,
    smoothing: int | None = None,
    log_scale: bool = True,
    title: str | None = None,
    legend_title: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot training curves.

    Accepts:
        - RunResult: single curve
        - list[RunResult]: averaged with CI
        - dict mapping labels to RunResult or list[RunResult]
    """
    if ax is None:
        _, ax = plt.subplots()

    # Normalize to dict[str, RunResult | list[RunResult]]
    if isinstance(data, RunResult):
        labeled_data = {None: data}
    elif isinstance(data, list):
        labeled_data = {None: data}
    else:
        labeled_data = data

    for label, series in labeled_data.items():
        _plot_series(ax, series, metric, label, ci, smoothing)

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel or metric)
    if any(k is not None for k in labeled_data.keys()):
        ax.legend(title=legend_title)
    if title:
        ax.set_title(title)

    return ax


# =============================================================================
# Specialized plotting functions
# =============================================================================


def plot_comparative(
    runs: RunResult | dict[str, RunResult | list[RunResult]],
    metric: str = "test_loss",
    ylabel: str | None = None,
    suffixes: tuple[str, str] = ("A", "B"),
    ci: float = 0.95,
    smoothing: int | None = None,
    log_scale: bool = True,
    title: str | None = None,
    legend_title: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot comparative training curves (model A vs B)."""
    if isinstance(runs, RunResult):
        runs = {None: runs}

    if ax is None:
        _, ax = plt.subplots()

    for name, data in runs.items():
        prefix = f"{name} " if name is not None else ""
        _plot_series(ax, data, f"{metric}_a", f"{prefix}{suffixes[0]}", ci, smoothing)
        _plot_series(ax, data, f"{metric}_b", f"{prefix}{suffixes[1]}", ci, smoothing)

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel or metric)
    ax.legend(title=legend_title)
    if title:
        ax.set_title(title)

    return ax


def plot_run(result: RunResult, metrics: list[str] | None = None) -> plt.Figure:
    """Quick visualization of a single run: loss + all tracked metrics."""
    if metrics is None:
        metrics = [m for m in result.metric_names() if m != "test_loss"]

    n_panels = 1 + len(metrics)
    n_cols = min(n_panels, 2)
    n_rows = math.ceil(n_panels / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
    )
    axes = axes.flatten()

    steps = result["step"]

    axes[0].plot(steps, result["test_loss"], label="test")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    for i, metric in enumerate(metrics):
        axes[i + 1].plot(steps, result[metric])
        axes[i + 1].set_xlabel("Step")
        axes[i + 1].set_ylabel(metric)

    for i in range(n_panels, len(axes)):
        axes[i].set_visible(False)

    fig.tight_layout()

    return fig


# =============================================================================
# Derived quantity helpers
# =============================================================================


def subtract_baseline(
    baseline: RunResult,
    runs: list[RunResult],
    metric: str = "test_loss",
) -> list[np.ndarray]:
    """Compute runs[i][metric] - baseline[metric] for each run."""
    base = np.array(baseline[metric])
    return [np.array(r[metric]) - base for r in runs]
