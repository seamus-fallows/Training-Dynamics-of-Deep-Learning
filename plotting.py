import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import stats

from dln.results import RunResult

plt.style.use("seaborn-v0_8-whitegrid")


def compute_ci(
    curves: list[list[float]], ci: float = 0.95
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


def plot(
    runs: dict[str, RunResult | list[RunResult]],
    metric: str = "train_loss",
    ci: float = 0.95,
    smoothing: int | None = None,
    log_scale: bool = True,
    title: str | None = None,
    legend_title: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots()

    for label, data in runs.items():
        if isinstance(data, RunResult):
            steps = np.asarray(data["step"])
            values = np.asarray(data[metric])

            if smoothing:
                values = smooth(values, smoothing)

            ax.plot(steps, values, label=label)

        else:
            steps = np.asarray(data[0]["step"])

            if len(data) == 1:
                # Single run, no CI
                values = np.asarray(data[0][metric])
                if smoothing:
                    values = smooth(values, smoothing)
                ax.plot(steps, values, label=label)
            else:
                # Multiple runs, compute CI
                curves = [run[metric] for run in data]
                mean, lower, upper = compute_ci(curves, ci)

                if smoothing:
                    mean = smooth(mean, smoothing)
                    lower = smooth(lower, smoothing)
                    upper = smooth(upper, smoothing)

                (line,) = ax.plot(steps, mean, label=label)
                ax.fill_between(steps, lower, upper, alpha=0.2, color=line.get_color())

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel(metric)
    ax.legend(title=legend_title)
    if title:
        ax.set_title(title)

    return ax


def plot_comparative(
    result: RunResult,
    metric: str = "train_loss",
    labels: tuple[str, str] = ("Model A", "Model B"),
    smoothing: int | None = None,
    log_scale: bool = True,
    title: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots()

    steps = np.asarray(result["step"])
    values_a = np.asarray(result[f"{metric}_a"])
    values_b = np.asarray(result[f"{metric}_b"])

    if smoothing:
        values_a = smooth(values_a, smoothing)
        values_b = smooth(values_b, smoothing)

    ax.plot(steps, values_a, label=labels[0])
    ax.plot(steps, values_b, label=labels[1])

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel(metric)
    ax.legend()
    if title:
        ax.set_title(title)

    return ax


def plot_run(result: RunResult, metrics: list[str] | None = None) -> Figure:
    if metrics is None:
        metrics = [m for m in result.metrics() if m not in ("train_loss", "test_loss")]

    has_test = result.has("test_loss")
    n_panels = 1 + len(metrics)
    n_cols = min(n_panels, 2)
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
    )
    axes = axes.flatten()

    steps = result["step"]

    axes[0].plot(steps, result["train_loss"], label="train")
    if has_test:
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


def auto_plot(result: RunResult, show: bool = True, save: bool = True) -> None:
    is_comparative = result.has("train_loss_a")

    if is_comparative:
        fig, ax = plt.subplots()
        plot_comparative(result, ax=ax)
    else:
        fig = plot_run(result)

    if save and result.output_dir:
        fig.savefig(result.output_dir / "plots.png", dpi=150, bbox_inches="tight")

    if not show:
        plt.close(fig)
