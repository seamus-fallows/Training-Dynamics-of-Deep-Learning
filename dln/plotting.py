import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy import stats

from .results import RunResult, SweepResult


# =============================================================================
# Core helpers
# =============================================================================


def compute_ci(
    curves: list[np.ndarray] | list[list[float]], ci: float = 0.95
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (mean, lower, upper) for confidence interval."""
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
        steps = np.asarray(series["step"])
        values = np.asarray(series[metric])
        if smoothing:
            values = smooth(values, smoothing)
        ax.plot(steps, values, label=label)
    elif len(series) == 1:
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
    | SweepResult
    | dict[str, RunResult | list[RunResult]],
    metric: str = "test_loss",
    ylabel: str | None = None,
    ci: float = 0.95,
    smoothing: int | None = None,
    log_scale: bool = True,
    title: str | None = None,
    legend_title: str | None = None,
    average: bool | str = False,
    ax: Axes | None = None,
) -> Axes:
    """Plot training curves.

    Accepts:
        - RunResult: single curve
        - list[RunResult]: averaged with CI
        - SweepResult: each param value as separate curve (or averaged if average=True)
        - dict mapping labels to RunResult or list[RunResult]

    Args:
        average: For SweepResult only. If True or a string, combine all runs into
            one curve with CI. String value becomes the label.
    """
    if ax is None:
        _, ax = plt.subplots()

    # Normalize to dict[str, RunResult | list[RunResult]]
    if isinstance(data, RunResult):
        labeled_data = {None: data}
    elif isinstance(data, SweepResult):
        if average:
            label = average if isinstance(average, str) else "averaged"
            labeled_data = {label: list(data.runs.values())}
        else:
            labeled_data = data.runs
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
    runs: RunResult | SweepResult | dict[str, RunResult | list[RunResult]],
    metric: str = "test_loss",
    suffixes: tuple[str, str] = ("A", "B"),
    ci: float = 0.95,
    smoothing: int | None = None,
    log_scale: bool = True,
    title: str | None = None,
    legend_title: str | None = None,
    average: bool | str = False,
    ax: Axes | None = None,
) -> Axes:
    """Plot comparative training curves (model A vs B)."""
    if isinstance(runs, RunResult):
        runs = {None: runs}
    elif isinstance(runs, SweepResult):
        if average:
            label = average if isinstance(average, str) else None
            runs = {label: list(runs.runs.values())}
        else:
            runs = runs.runs

    if ax is None:
        _, ax = plt.subplots()

    for name, data in runs.items():
        prefix = f"{name} " if name is not None else ""

        if isinstance(data, RunResult):
            steps = np.asarray(data["step"])
            values_a = np.asarray(data[f"{metric}_a"])
            values_b = np.asarray(data[f"{metric}_b"])

            if smoothing:
                values_a = smooth(values_a, smoothing)
                values_b = smooth(values_b, smoothing)

            ax.plot(steps, values_b, label=f"{prefix}{suffixes[1]}")
            ax.plot(steps, values_a, label=f"{prefix}{suffixes[0]}")

        else:
            steps = np.asarray(data[0]["step"])

            if len(data) == 1:
                values_a = np.asarray(data[0][f"{metric}_a"])
                values_b = np.asarray(data[0][f"{metric}_b"])

                if smoothing:
                    values_a = smooth(values_a, smoothing)
                    values_b = smooth(values_b, smoothing)

                ax.plot(steps, values_b, label=f"{prefix}{suffixes[1]}")
                ax.plot(steps, values_a, label=f"{prefix}{suffixes[0]}")

            else:
                curves_a = [r[f"{metric}_a"] for r in data]
                curves_b = [r[f"{metric}_b"] for r in data]

                mean_a, lower_a, upper_a = compute_ci(curves_a, ci)
                mean_b, lower_b, upper_b = compute_ci(curves_b, ci)

                if smoothing:
                    mean_a = smooth(mean_a, smoothing)
                    lower_a = smooth(lower_a, smoothing)
                    upper_a = smooth(upper_a, smoothing)
                    mean_b = smooth(mean_b, smoothing)
                    lower_b = smooth(lower_b, smoothing)
                    upper_b = smooth(upper_b, smoothing)

                (line_b,) = ax.plot(steps, mean_b, label=f"{prefix}{suffixes[1]}")
                ax.fill_between(
                    steps, lower_b, upper_b, alpha=0.2, color=line_b.get_color()
                )
                (line_a,) = ax.plot(steps, mean_a, label=f"{prefix}{suffixes[0]}")
                ax.fill_between(
                    steps, lower_a, upper_a, alpha=0.2, color=line_a.get_color()
                )

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel(metric)
    ax.legend(title=legend_title)
    if title:
        ax.set_title(title)

    return ax


def plot_run(result: RunResult, metrics: list[str] | None = None) -> None:
    """Quick visualization of a single run: loss + all tracked metrics."""
    if metrics is None:
        metrics = [m for m in result.metric_names() if m != "test_loss"]

    n_panels = 1 + len(metrics)
    n_cols = min(n_panels, 2)
    n_rows = (n_panels + n_cols - 1) // n_cols

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


def auto_plot(result: RunResult, show: bool = True) -> None:
    """Auto-generate plots for a run."""
    is_comparative = result.has("test_loss_a")

    if is_comparative:
        fig, ax = plt.subplots()
        plot_comparative(result, ax=ax)
    else:
        plot_run(result)
        fig = plt.gcf()

    if result.output_dir:
        fig.savefig(result.output_dir / "plots.png", dpi=150, bbox_inches="tight")

    if not show:
        plt.close(fig)


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
