"""Experiment Examples"""

# %%
import shutil
from pathlib import Path
from notebook_utils import (
    run_single,
    run_comparative,
    run_sweep,
    plot_metrics,
    plot_comparative,
    plot_sweep,
    run_grid_search,
)

# %% Example 1: Basic Single Run
history = run_single("example_1", "diagonal_teacher")
plot_metrics(history, ["train_loss"])

# %% Example 2: Single Run with Metrics
history = run_single(
    "example_2", "diagonal_teacher", overrides=["metrics=[weight_norm,gradient_norm]"]
)
plot_metrics(history, ["train_loss", "weight_norm", "gradient_norm"])

# %% Example 3: Learning Rate Sweep
results = run_sweep(
    "example_3", "diagonal_teacher", "training.lr", [0.0005, 0.001, 0.002]
)
plot_sweep(results)

# %% Example 4: Comparative Training
history = run_comparative(
    "example_4", "diagonal_teacher", overrides=["training_b.batch_size=10"]
)
plot_comparative(history)

# %% Example 5: Comparative with Metrics
history = run_comparative(
    "example_5",
    "diagonal_teacher",
    overrides=[
        "training_b.batch_size=10",
        "model_metrics=[weight_norm,gradient_norm]",
        "comparative_metrics=[param_distance,param_cosine_sim]",
    ],
)
plot_comparative(history, comparative_metrics=["param_distance", "param_cosine_sim"])
plot_metrics(history, ["weight_norm", "gradient_norm"])

# %% Example 6: Different Seeds
history = run_comparative(
    "example_6",
    "diagonal_teacher",
    overrides=["training_a.model_seed=42", "training_b.model_seed=999"],
)
plot_comparative(history)

# %% Example 7a: Different Dataset Type (using config)
history = run_single(
    "example_7a",
    "random_teacher",
    overrides=["training.lr=0.04"],
)
plot_metrics(history, ["train_loss"])

# %% Example 7b: Different Dataset Type (using override)
history = run_single(
    "example_7b",
    "diagonal_teacher",
    overrides=[
        "data.type=random_teacher",
        "++data.params={mean: 0.0, std: 1.0}",
        "training.lr=0.04",
    ],
)
plot_metrics(history, ["train_loss"])

# %% Example 8: Comparing Optimizers
history = run_comparative(
    "example_8",
    "diagonal_teacher",
    overrides=["training_a.optimizer=SGD", "training_b.optimizer=Adam"],
)
plot_comparative(history)

# %% Example 9: Architecture Sweep (Depth)
results = run_sweep("example_9", "diagonal_teacher", "model.num_hidden", [1, 2, 3])
plot_sweep(results)

# %% Example 10: Comparing Initialization Scaling (Gamma)
history = run_comparative(
    "example_10",
    "diagonal_teacher",
    overrides=["shared.model.gamma=1.2", "model_b.gamma=1.5"],
)
plot_comparative(history)

# %% Example 11: Tracking Generalization with Test Split
history = run_single(
    "example_11",
    "diagonal_teacher",
    overrides=["data.test_split=0.2", "data.num_samples=100"],
)
plot_metrics(history, ["train_loss", "test_loss"], combine=True)

# %% Example 12: Grid Search (LR x Depth)
results = run_grid_search(
    "example_12",
    "diagonal_teacher",
    params={"training.lr": [0.0005, 0.001], "model.num_hidden": [2, 3, 4]},
)
plot_sweep(results)

# %% Snapping Experiment

# High noise baseline
history_high = run_single(
    "snap_high", "diagonal_teacher", overrides=["training.batch_size=10"]
)

# Low noise baseline
history_low = run_single(
    "snap_low", "diagonal_teacher", overrides=["training.batch_size=null"]
)

# Switch from high to low noise at step 1000
history_switch = run_single(
    "snap_switch",
    "diagonal_teacher",
    overrides=[
        "training.batch_size=10",
        "switch.step=1000",
        "switch.batch_size=null",
    ],
)

# Plot together
results = {
    "high_noise": history_high,
    "low_noise": history_low,
    "switch": history_switch,
}
# %%
# Without smoothing
plot_sweep(results, title="Snapping Experiment")

# With smoothing (50-step moving average)
plot_sweep(results, title="Snapping Experiment", smoothing=20)
# %%
# Run to delete outputs
shutil.rmtree(Path("outputs/notebook"), ignore_errors=True)

# %%
