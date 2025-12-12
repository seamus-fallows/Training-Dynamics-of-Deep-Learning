"""
Experiment examples using the runner and plotting utilities.
"""

# %%
from pathlib import Path

from runner import (
    run,
    run_comparative,
    run_sweep,
    run_seeds,
    load_run,
    load_hydra_sweep,
)
from plotting import plot, plot_run, plot_comparative

# =============================================================================
# Example 1: Basic Single Run
# =============================================================================
# %%

result = run("diagonal_teacher")
plot({"diagonal_teacher": result})

# =============================================================================
# Example 2: Single Run with Metrics
# =============================================================================
# %%

result = run(
    "diagonal_teacher",
    overrides={
        "metrics": ["weight_norm", "grad_norm_squared"],
        "metric_data.mode": "population",
    },
)
plot_run(result)

# =============================================================================
# Example 3: Learning Rate Sweep
# =============================================================================
# %%

sweep = run_sweep(
    "diagonal_teacher", param="training.lr", values=[0.0005, 0.001, 0.002]
)
plot(sweep.runs, legend_title="lr")

# =============================================================================
# Example 4: Comparative Training
# =============================================================================
# %%

result = run_comparative("diagonal_teacher", overrides={"training_b.batch_size": 10})
plot_comparative(result, labels=("Full batch", "Batch=10"))

# =============================================================================
# Example 5: Comparative with Metrics
# =============================================================================
# %%

result = run_comparative(
    "diagonal_teacher",
    overrides={
        "training_b.batch_size": 10,
        "model_metrics": ["weight_norm", "grad_norm_squared"],
        "comparative_metrics": ["param_distance", "param_cosine_sim"],
    },
)
plot_comparative(result, labels=("Full batch", "Batch=10"))
plot_comparative(
    result, metric="weight_norm", labels=("Full batch", "Batch=10"), log_scale=False
)
plot({"param_distance": result}, metric="param_distance", log_scale=False)

# =============================================================================
# Example 6: Different Model Seeds
# =============================================================================
# %%

result = run_comparative(
    "diagonal_teacher", overrides={"model_a.seed": 42, "model_b.seed": 999}
)
plot_comparative(result, labels=("seed=42", "seed=999"))

# =============================================================================
# Example 7: Different Dataset Type
# =============================================================================
# %%

result = run("random_teacher", overrides={"training.lr": 0.04})
plot({"random_teacher": result})

# =============================================================================
# Example 8: Comparing Optimizers
# =============================================================================
# %%

result = run_comparative(
    "diagonal_teacher",
    overrides={"training_a.optimizer": "SGD", "training_b.optimizer": "Adam"},
)
plot_comparative(result, labels=("SGD", "Adam"))

# =============================================================================
# Example 9: Architecture Sweep (Depth)
# =============================================================================
# %%

sweep = run_sweep("diagonal_teacher", param="model.num_hidden", values=[1, 2, 3])
plot(sweep.runs, legend_title="num_hidden")

# =============================================================================
# Example 10: Comparing Initialization Scaling (Gamma)
# =============================================================================
# %%

result = run_comparative(
    "diagonal_teacher",
    overrides={"shared.model.gamma": 1.2, "model_b.gamma": 1.5},
)
plot_comparative(result, labels=("gamma=1.2", "gamma=1.5"))

# =============================================================================
# Example 11: Tracking Generalization with Test Split
# =============================================================================
# %%

result = run(
    "diagonal_teacher", overrides={"data.test_split": 0.2, "data.num_samples": 100}
)
plot_run(result)

# =============================================================================
# Example 12: Multi-Seed Averaging with CI
# =============================================================================
# %%

results = run_seeds("diagonal_teacher", n_seeds=5)
plot({"diagonal_teacher": results})

# =============================================================================
# Example 13: Sweep with Multiple Seeds
# =============================================================================
# %%

sweep = run_sweep(
    "diagonal_teacher", param="training.lr", values=[0.0005, 0.001], n_seeds=3
)
plot(sweep.runs, legend_title="lr")

# =============================================================================
# Example 14: Mixing Single Run and Averaged Runs
# =============================================================================
# %%

gd_baseline = run("diagonal_teacher", overrides={"training.batch_size": None})
sgd_runs = run_seeds(
    "diagonal_teacher", n_seeds=5, overrides={"training.batch_size": 10}
)

plot({"GD (full batch)": gd_baseline, "SGD (batch=10)": sgd_runs})

# =============================================================================
# Example 15: Snapping Experiment (High to Low Noise)
# =============================================================================
# %%

common_overrides = {
    "model.hidden_dim": 150,
    "model.seed": 2,
    "data.num_samples": 500,
    "max_steps": 4000,
    "training.lr": 0.0006,
}

result_low = run(
    "diagonal_teacher", overrides={**common_overrides, "training.batch_size": None}
)
result_switch = run(
    "diagonal_teacher",
    overrides={
        **common_overrides,
        "training.batch_size": 1,
        "callbacks": [
            {"name": "switch_batch_size", "params": {"step": 1100, "batch_size": None}}
        ],
    },
)

plot({"low_noise": result_low, "high_to_low": result_switch}, title="High to Low Noise")
plot(
    {"low_noise": result_low, "high_to_low": result_switch},
    title="Smoothed",
    smoothing=40,
)

# =============================================================================
# Example 16: Snapping Experiment (Low to High Noise)
# =============================================================================
# %%

result_switch = run(
    "diagonal_teacher",
    overrides={
        **common_overrides,
        "training.batch_size": None,
        "callbacks": [
            {"name": "switch_batch_size", "params": {"step": 1100, "batch_size": 1}}
        ],
    },
)

plot({"low_noise": result_low, "low_to_high": result_switch}, title="Low to High Noise")
plot(
    {"low_noise": result_low, "low_to_high": result_switch},
    title="Smoothed",
    smoothing=40,
)

# =============================================================================
# Example 17: Loading Saved Results
# =============================================================================
# %%

# result = load_run(Path("outputs/runs/diagonal_teacher_2024-01-15_10-30-00"))
# plot_run(result)

# sweep = load_hydra_sweep(Path("outputs/single/multiruns/..."), sweep_param="training.batch_size")
# plot(sweep.runs)
