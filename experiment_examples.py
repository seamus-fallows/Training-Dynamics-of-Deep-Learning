"""
Experiment examples
"""

# %%
import matplotlib.pyplot as plt

from runner import run, run_comparative, run_sweep, run_comparative_sweep
from plotting import plot, plot_run, plot_comparative

# =============================================================================
# Example 1: Basic Single Run
# =============================================================================
# %%
# In yaml we have show plotting enabled by default, so this will plot automatically
result = run("diagonal_teacher")


# =============================================================================
# Example 2: Run with Metrics
# =============================================================================
# %%

result = run(
    "diagonal_teacher",
    overrides={
        "metrics": ["weight_norm", "grad_norm_squared"],
        "metric_data.mode": "population",
    },
)
# =============================================================================
# Example 3: Sweep over Learning Rate
# =============================================================================
# %%

sweep = run_sweep(
    "diagonal_teacher",
    param="training.lr",
    values=[0.0005, 0.001, 0.002],
)
plot(sweep, legend_title="lr")

# =============================================================================
# Example 4: Batch Seed Sweep (Averaged) + Full-Batch Baseline
# =============================================================================
# %%
overrides = {
    "training.batch_size": 50,
    "training.lr": 0.0002,
    "max_steps": 5000,
}

sweep = run_sweep(
    "diagonal_teacher",
    param="training.batch_seed",
    values=range(5),
    overrides=overrides,
)
# %%
baseline = run(
    "diagonal_teacher",
    autoplot=False,
    overrides={**overrides, "training.batch_size": None},
)

# Averaged with CI
plot({**sweep.to_average("SGD (batch=20)"), "GD (full batch)": baseline})

# =============================================================================
# Example 5: Comparative Run (Different Batch Sizes)
# =============================================================================
# %%

result = run_comparative(
    "diagonal_teacher",
    overrides={
        "training_b.batch_size": 10,
        "comparative_metrics": ["param_distance"],
    },
    autoplot=False,
)
plot_comparative(result, suffixes=("Full batch", "Batch=10"))
plot(result, metric="param_distance", log_scale=False)

# =============================================================================
# Example 6: Comparative Sweep (Averaged with CI)
# =============================================================================
# %%

# model_a is considered the baseline and will be plotted second so it appears on top
sweep = run_comparative_sweep(
    "diagonal_teacher",
    param="shared.training.batch_seed",
    values=range(5),
    overrides={
        "training_b.batch_size": 10,
        "comparative_metrics": ["param_distance"],
    },
)
plot_comparative(sweep.to_average(), suffixes=("Full batch", "Batch=10"))
plot(sweep.to_average(), metric="param_distance", log_scale=False)

# =============================================================================
# Example 7: Comparative with Model Metrics
# =============================================================================
# %%

result = run_comparative(
    "diagonal_teacher",
    overrides={
        "training_b.batch_size": 10,
        "model_metrics": ["weight_norm"],
        "comparative_metrics": ["param_distance"],
        "metric_data.mode": "population",
    },
    autoplot=False,
)
plot_comparative(result, suffixes=("Full batch", "Batch=10"))
plot_comparative(
    result, metric="weight_norm", suffixes=("Full batch", "Batch=10"), log_scale=False
)
plot(result, metric="param_distance", log_scale=False)

# =============================================================================
# Example 8: Comparative Sweep over Shared Parameter
# =============================================================================
# %%

sweep = run_comparative_sweep(
    "diagonal_teacher",
    param="shared.training.lr",
    values=[0.0005, 0.001],
    overrides={"training_b.batch_size": 20, "max_steps": 3000},
)

for lr, result in sweep.runs.items():
    plot_comparative(result, suffixes=("Full batch", "Batch=20"), title=f"lr={lr}")

# =============================================================================
# Example 9: Different Config File
# =============================================================================
# %%

run("random_teacher", overrides={"training.lr": 0.04})

# =============================================================================
# Example 10: Comparing Optimizers
# =============================================================================
# %%

results = run_comparative(
    "diagonal_teacher",
    overrides={
        "training_a.optimizer": "SGD",
        "training_b.optimizer": "Adam",
        "max_steps": 2000,
        "shared.training.lr": 0.001,
    },
)

# =============================================================================
# Example 11: Train/Test Split
# =============================================================================
# %%

results = run(
    "diagonal_teacher", overrides={"data.test_split": 0.2, "data.num_samples": 125}
)

# =============================================================================
# Example 12: Batch Size Switching (Snapping)
# =============================================================================
# %%

common = {
    "model.hidden_dim": 50,
    "model.seed": 2,
    "data.num_samples": 200,
    "max_steps": 5000,
    "training.lr": 0.0005,
}

high_to_low = run(
    "diagonal_teacher",
    overrides={
        **common,
        "training.batch_size": 5,
        "callbacks": [
            {"name": "switch_batch_size", "params": {"step": 1100, "batch_size": None}}
        ],
    },
    autoplot=False,
)
baseline = run(
    "diagonal_teacher",
    overrides={**common, "training.batch_size": None},
    autoplot=False,
)

plot(
    {"high_to_low": high_to_low, "baseline": baseline},
    smoothing=0,
    title="High to Low",
)

low_to_high = run(
    "diagonal_teacher",
    overrides={
        **common,
        "training.batch_size": None,
        "callbacks": [
            {"name": "switch_batch_size", "params": {"step": 1100, "batch_size": 5}}
        ],
    },
    autoplot=False,
)

plot(
    {"low_to_high": low_to_high, "baseline": baseline},
    smoothing=0,
    title="Low to High",
)

# =============================================================================
# Example 14: Snapping Averaged over Batch Seeds
# =============================================================================
# %%

common = {
    "model.hidden_dim": 50,
    "model.seed": 2,
    "data.num_samples": 100,
    "max_steps": 5000,
    "training.lr": 0.0002,
}

sweep = run_sweep(
    "diagonal_teacher",
    param="training.batch_seed",
    values=range(5),
    max_workers=5,
    overrides={
        **common,
        "callbacks": [
            {"name": "switch_batch_size", "params": {"step": 1100, "batch_size": 20}}
        ],
    },
)
baseline = run(
    "diagonal_teacher",
    overrides={**common, "training.batch_size": None},
    autoplot=False,
)

plot(
    {**sweep.to_average("high_to_low"), "baseline": baseline},
    title="High to Low (Averaged)",
)

# =============================================================================
# Example 15: Plotting Options
# =============================================================================
# %%

result = run("diagonal_teacher", autoplot=False)

# Smoothing
plot(result, smoothing=50, title="Smoothed")

# Linear scale
plot(result, log_scale=False, title="Linear Scale")

# Legend title
sweep = run_sweep("diagonal_teacher", param="training.lr", values=[0.0005, 0.001])
plot(sweep, legend_title="Learning Rate")

# Multiple subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
plot(result, ax=axes[0], title="Log Scale")
plot(result, ax=axes[1], log_scale=False, title="Linear Scale")
fig.tight_layout()

# =============================================================================
# Example 16: Loading Saved Results
# =============================================================================
# %%

# from pathlib import Path
# from runner import load_run, load_hydra_sweep
#
# result = load_run(Path("outputs/runs/diagonal_teacher_2024-01-15_10-30-00"))
# plot(result)
#
# sweep = load_hydra_sweep(Path("outputs/sweeps/..."), sweep_param="training.lr")
# plot(sweep)
