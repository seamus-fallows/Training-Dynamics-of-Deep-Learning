"""
Experiment Examples for Deep Linear Network Training

This script demonstrates how to run experiments IN-PROCESS.
This is superior to subprocess calls because:
1. Progress bars (tqdm) render correctly.
2. Debugging is easier (you see stack traces).
3. You have direct access to results without path searching.

Usage: Run cells interactively in VSCode/Jupyter.
"""

# %% [markdown]
# # 1. Setup and Imports

# %%
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import matplotlib.pyplot as plt
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

# Import your actual training functions
# Note: Ensure these functions accept 'output_dir' as an argument!
from run import run_experiment
from run_comparative import run_comparative_experiment


# %% [markdown]
# # 2. Helper Functions


# %%
def get_config(
    config_path: str, config_name: str, overrides: Optional[List[str]] = None
) -> DictConfig:
    """
    Safely loads a Hydra config without crashing if already initialized.
    """
    # Clear any existing Hydra instance to avoid conflicts in notebooks
    GlobalHydra.instance().clear()

    initialize(version_base=None, config_path=config_path)
    return compose(config_name=config_name, overrides=overrides or [])


def load_history(output_dir: Path) -> List[Dict[str, Any]]:
    """Loads the training history from the JSONL file in the output directory."""
    history_path = output_dir / "history.jsonl"
    if not history_path.exists():
        raise FileNotFoundError(f"No history found at {history_path}")

    data = []
    with history_path.open("r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def plot_history(
    history: List[Dict[str, Any]], title: str = "Training History"
) -> None:
    steps = [h["step"] for h in history]
    train_loss = [h["train_loss"] for h in history]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_loss, label="Train Loss", linewidth=2)

    # Plot test loss if available
    if "test_loss" in history[0] and history[0]["test_loss"] is not None:
        test_loss = [h["test_loss"] for h in history]
        plt.plot(steps, test_loss, label="Test Loss", linewidth=2, linestyle="--")

    plt.yscale("log")
    plt.xlabel("Steps")
    plt.ylabel("Loss (MSE)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_comparative(
    history: List[Dict[str, Any]], title: str = "Comparative Training"
) -> None:
    steps = [h["step"] for h in history]
    loss_a = [h["train_loss_a"] for h in history]
    loss_b = [h["train_loss_b"] for h in history]
    dist = [h["param_distance"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Loss Plot
    ax1.plot(steps, loss_a, label="Model A", alpha=0.8)
    ax1.plot(steps, loss_b, label="Model B", alpha=0.8)
    ax1.set_yscale("log")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Train Loss")
    ax1.set_title("Loss Trajectories")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Distance Plot
    ax2.plot(steps, dist, color="green", label="Param Distance")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("L2 Distance")
    ax2.set_title("Distance between Model A and B")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_sweep(
    results: Dict[str, List[Dict[str, Any]]], title: str = "Sweep Results"
) -> None:
    plt.figure(figsize=(10, 6))

    for label, history in results.items():
        steps = [h["step"] for h in history]
        loss = [h["train_loss"] for h in history]
        plt.plot(steps, loss, label=label, alpha=0.8)

    plt.yscale("log")
    plt.xlabel("Steps")
    plt.ylabel("Train Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# %% [markdown]
# # Example 1: Basic Single Run
# Runs the model with default parameters from `configs/single/diagonal_teacher.yaml`.

# %%
# 1. Setup Config
cfg = get_config(
    "configs/single",
    "diagonal_teacher",
    overrides=[
        "training.lr=0.001",  # Higher learning rate
        "model.hidden_size=50",  # Wider layers
    ],
)

# 2. Define explicit output path (No searching required!)
output_dir = Path("outputs/notebook/example_1_basic")
output_dir.mkdir(parents=True, exist_ok=True)

# 3. Run
print(f"Starting training in {output_dir}...")
run_experiment(cfg, output_dir=output_dir)

# 4. Analyze
history = load_history(output_dir)
print(f"Final Loss: {history[-1]['train_loss']:.4f}")
plot_history(history, title="Example 1: Basic Run")


# %% [markdown]
# # Example 2: Run with Overrides
# Changing hyperparameters (Learning Rate & Model Size) programmatically.

# %%
cfg = get_config(
    "configs/single",
    "diagonal_teacher",
    overrides=[
        "training.lr=0.001",  # Higher learning rate
        "model.hidden_size=50",  # Wider layers
    ],
)

output_dir = Path("outputs/notebook/example_2_overrides")
output_dir.mkdir(parents=True, exist_ok=True)

run_experiment(cfg, output_dir=output_dir)
history = load_history(output_dir)

plot_history(history, title="Example 2: LR=0.01, Width=50")


# %% [markdown]
# # Example 3: Learning Rate Sweep (Python Loop)
# Instead of 'multirun', we just loop in Python. It's simpler and more flexible.

# %%
learning_rates = [0.0001, 0.001, 0.01]
results = {}

for lr in learning_rates:
    print(f"\n--- Running Sweep: LR = {lr} ---")

    # reload config for each run to be safe
    cfg = get_config(
        "configs/single",
        "diagonal_teacher",
        overrides=[f"training.lr={lr}", "training.max_steps=1500"],
    )

    output_dir = Path(f"outputs/notebook/example_3_sweep/lr_{lr}")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_experiment(cfg, output_dir=output_dir)
    results[f"LR={lr}"] = load_history(output_dir)

plot_sweep(results, title="Learning Rate Sweep")


# %% [markdown]
# # Example 4: Comparative Training
# Training two models simultaneously to compare trajectories.

# %%
cfg = get_config(
    "configs/comparative",
    "diagonal_teacher",
    overrides=[
        "max_steps=3000",
        "shared.training.lr=0.0005",  # Updates both models
        "training_b.batch_size=10",  # Model B uses mini-batches, A uses full-batch (default)
    ],
)

output_dir = Path("outputs/notebook/example_4_comparative")
output_dir.mkdir(parents=True, exist_ok=True)

run_comparative_experiment(cfg, output_dir=output_dir)

history = load_history(output_dir)
plot_comparative(history, title="Batch Size: Full (A) vs 10 (B)")


# %% [markdown]
# # Example 5: Comparative with Different Seeds
# Checking sensitivity to initialization.

# %%
cfg = get_config(
    "configs/comparative",
    "diagonal_teacher",
    overrides=[
        "training_a.model_seed=42",
        "training_b.model_seed=999",
        "shared.training.lr=0.001",
    ],
)

output_dir = Path("outputs/notebook/example_5_seeds")
output_dir.mkdir(parents=True, exist_ok=True)

run_comparative_experiment(cfg, output_dir=output_dir)

history = load_history(output_dir)
plot_comparative(history, title="Effect of Random Seed")
# %%
