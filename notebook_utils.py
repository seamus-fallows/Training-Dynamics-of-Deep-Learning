"""
Notebook utilities for running and visualizing experiments.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from run import run_experiment
from run_comparative import run_comparative_experiment

# ============================================
# Config & History
# ============================================

OUTPUT_ROOT = Path("outputs/notebook")


def get_config(
    config_path: str, config_name: str, overrides: Optional[List[str]] = None
) -> DictConfig:
    """Safely loads a Hydra config."""
    GlobalHydra.instance().clear()
    initialize(version_base=None, config_path=config_path)
    return compose(config_name=config_name, overrides=overrides or [])


# ============================================
# Run Experiments
# ============================================


def run_single(
    name: str,
    config_name: str,
    overrides: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Run a single experiment and return history."""
    cfg = get_config("configs/single", config_name, overrides)
    output_dir = OUTPUT_ROOT / name
    output_dir.mkdir(parents=True, exist_ok=True)
    return run_experiment(cfg, output_dir=output_dir)


def run_comparative(
    name: str,
    config_name: str,
    overrides: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Run a comparative experiment and return history."""
    cfg = get_config("configs/comparative", config_name, overrides)
    output_dir = OUTPUT_ROOT / name
    output_dir.mkdir(parents=True, exist_ok=True)
    return run_comparative_experiment(cfg, output_dir=output_dir)


def run_sweep(
    name: str,
    config_name: str,
    param: str,
    values: List[Any],
    overrides: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run a parameter sweep and return dict of histories."""
    results = {}
    for val in values:
        print(f"\n--- {param}={val} ---")
        sweep_overrides = (overrides or []) + [f"{param}={val}"]
        cfg = get_config("configs/single", config_name, sweep_overrides)
        output_dir = OUTPUT_ROOT / name / f"{param}_{val}"
        output_dir.mkdir(parents=True, exist_ok=True)
        results[f"{param}={val}"] = run_experiment(cfg, output_dir=output_dir)
    return results


def run_grid_search(
    name: str,
    config_name: str,
    params: dict[str, List[Any]],
    overrides: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run a grid search over multiple parameters."""
    from itertools import product

    results = {}
    param_names = list(params.keys())
    param_values = list(params.values())

    for combo in product(*param_values):
        label = ", ".join(f"{p}={v}" for p, v in zip(param_names, combo))
        combo_overrides = (overrides or []) + [
            f"{p}={v}" for p, v in zip(param_names, combo)
        ]

        print(f"\n--- {label} ---")
        cfg = get_config("configs/single", config_name, combo_overrides)

        dir_name = "_".join(
            f"{p.split('.')[-1]}={v}" for p, v in zip(param_names, combo)
        )
        output_dir = OUTPUT_ROOT / name / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        results[label] = run_experiment(cfg, output_dir=output_dir)

    return results


# ============================================
# Plotting
# ============================================


def smooth(values: List[float], window: int = 50) -> List[float]:
    """Simple moving average smoothing."""
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed.append(sum(values[start : i + 1]) / (i - start + 1))
    return smoothed


def smooth_centered(values: List[float], window: int = 50) -> List[float]:
    smoothed = []
    half = window // 2
    for i in range(len(values)):
        start = max(0, i - half)
        end = min(len(values), i + half + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    return smoothed


def plot_metrics(
    history: List[Dict[str, Any]],
    metrics: List[str],
    title: Optional[str] = None,
    log_scale: bool = True,
    combine: bool = False,
) -> None:
    """Plot metrics from history. Set combine=True to plot all on one graph."""
    steps = [h["step"] for h in history]

    if combine:
        plt.figure(figsize=(8, 5))
        for metric in metrics:
            if metric in history[0]:
                plt.plot(steps, [h[metric] for h in history], label=metric)
        if log_scale:
            plt.yscale("log")
        plt.xlabel("Steps")
        plt.legend()
        plt.grid(True, alpha=0.3)
        if title:
            plt.title(title)
        plt.show()
        return

    grouped = []
    for metric in metrics:
        a_key = f"{metric}_a"
        b_key = f"{metric}_b"
        if a_key in history[0] and b_key in history[0]:
            grouped.append((a_key, b_key))
        elif metric in history[0]:
            grouped.append((metric,))

    if not grouped:
        print("No valid metrics found")
        return

    n_plots = len(grouped)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for i, group in enumerate(grouped):
        for metric in group:
            values = [h[metric] for h in history]
            axes[i].plot(steps, values, label=metric, linewidth=2)
        if log_scale:
            axes[i].set_yscale("log")
        axes[i].set_xlabel("Steps")
        axes[i].set_title(group[0].replace("_a", ""))
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    if title:
        plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_comparative(
    history: List[Dict[str, Any]],
    title: str = "Comparative Training",
    comparative_metrics: Optional[List[str]] = None,
) -> None:
    """Plot comparative training results."""
    steps = [h["step"] for h in history]
    loss_a = [h["train_loss_a"] for h in history]
    loss_b = [h["train_loss_b"] for h in history]

    if comparative_metrics:
        comparative_metrics = [m for m in comparative_metrics if m in history[0]]

    n_metric_plots = len(comparative_metrics) if comparative_metrics else 0
    n_plots = 1 + n_metric_plots

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    axes[0].plot(steps, loss_a, label="Model A", alpha=0.8)
    axes[0].plot(steps, loss_b, label="Model B", alpha=0.8)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Train Loss")
    axes[0].set_title("Loss Trajectories")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for i, metric in enumerate(comparative_metrics or []):
        values = [h[metric] for h in history]
        axes[i + 1].plot(steps, values, color="green")
        axes[i + 1].set_xlabel("Steps")
        axes[i + 1].set_ylabel(metric)
        axes[i + 1].set_title(metric)
        axes[i + 1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_sweep(
    results: Dict[str, List[Dict[str, Any]]],
    metric: str = "train_loss",
    title: str = "Sweep Results",
    log_scale: bool = True,
    smoothing: int | None = None,
) -> None:
    """Plot a single metric across multiple runs.

    Args:
        results: Dict mapping labels to history lists.
        metric: Which metric to plot.
        title: Plot title.
        log_scale: Whether to use log scale on y-axis.
        smoothing: If provided, SMA window size. Shows smoothed curves alongside faded raw data.
    """
    plt.figure(figsize=(10, 6))

    for label, history in results.items():
        steps = [h["step"] for h in history]
        values = [h[metric] for h in history]

        if smoothing is not None:
            plt.plot(steps, smooth(values, smoothing), label=label, linewidth=2)
        else:
            plt.plot(steps, values, label=label, alpha=0.8)

    if log_scale:
        plt.yscale("log")
    plt.xlabel("Steps")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
