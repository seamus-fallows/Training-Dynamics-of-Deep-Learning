# %%
"""
Verify that metrics are consistent across runs with different max_steps.
For each configuration, we run 8000, 10000, and 12000 steps and check
that all metrics agree at overlapping steps.
"""

import numpy as np
from itertools import product
from runner import run
import matplotlib.pyplot as plt

# %%
# Configuration
WIDTHS = [10, 100]
NOISE_LEVELS = [0.0, 0.2]
GAMMA = 1.0
MAX_STEPS_LIST = [8000, 10000, 12000]
EVALUATE_EVERY = 20

common_overrides = {
    "model.gamma": GAMMA,
    "model.num_hidden": 3,
    "data.train_samples": 500,
    "data.test_samples": 500,
    "training.lr": 0.0001,
    "training.batch_size": 50,
    "training.batch_seed": 0,
    "model.seed": 0,
    "data.data_seed": 0,
    "evaluate_every": EVALUATE_EVERY,
    "metrics": ["trace_covariances"],
    "metric_data.mode": "population",
    "plotting.enabled": False,
}

# %%
# Run experiments and check consistency

all_passed = True
all_results = {}

for width, noise in product(WIDTHS, NOISE_LEVELS):
    print(f"\n{'=' * 60}")
    print(f"Width={width}, Noise={noise}")
    print("=" * 60)

    overrides = {
        **common_overrides,
        "model.hidden_dim": width,
        "data.noise_std": noise,
    }

    # Run for each max_steps
    results = {}
    for max_steps in MAX_STEPS_LIST:
        print(f"Running max_steps={max_steps}...")
        results[max_steps] = run(
            "diagonal_teacher",
            overrides={**overrides, "max_steps": max_steps},
            autoplot=False,
        )

    all_results[(width, noise)] = results

    # Compare overlapping steps
    min_steps = min(MAX_STEPS_LIST)
    baseline = results[min_steps]
    baseline_steps = np.array(baseline["step"])

    metrics_to_check = [
        "train_loss",
        "test_loss",
        "grad_norm_squared",
        "trace_gradient_covariance",
        "trace_hessian_covariance",
    ]

    for max_steps in MAX_STEPS_LIST:
        if max_steps == min_steps:
            continue

        result = results[max_steps]
        result_steps = np.array(result["step"])

        # Find overlapping indices
        n_overlap = len(baseline_steps)

        print(
            f"\nComparing {min_steps} vs {max_steps} (first {n_overlap} evaluations):"
        )

        for metric in metrics_to_check:
            baseline_vals = np.array(baseline[metric])
            result_vals = np.array(result[metric])[:n_overlap]

            if np.allclose(baseline_vals, result_vals, rtol=1e-10, atol=1e-12):
                print(f"  {metric}: ✓ MATCH")
            else:
                all_passed = False
                diff_indices = np.where(
                    ~np.isclose(baseline_vals, result_vals, rtol=1e-10, atol=1e-12)
                )[0]
                print(
                    f"  {metric}: ✗ MISMATCH at steps {baseline_steps[diff_indices[:5]]}..."
                )

                # Show first mismatch details
                idx = diff_indices[0]
                print(
                    f"    Step {baseline_steps[idx]}: {baseline_vals[idx]} vs {result_vals[idx]}"
                )
                print(f"    Diff: {abs(baseline_vals[idx] - result_vals[idx])}")

# %%
print(f"\n{'=' * 60}")
if all_passed:
    print("ALL CHECKS PASSED ✓")
else:
    print("SOME CHECKS FAILED ✗")
print("=" * 60)

# %%
# Plot results for visual inspection

metrics_to_plot = [
    "train_loss",
    "grad_norm_squared",
    "trace_gradient_covariance",
    "trace_hessian_covariance",
]

for width, noise in product(WIDTHS, NOISE_LEVELS):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics_to_plot):
        for max_steps in MAX_STEPS_LIST:
            result = all_results[(width, noise)][max_steps]
            steps = result["step"]
            values = result[metric]
            ax.plot(steps, values, label=f"max_steps={max_steps}", alpha=0.7)

        ax.set_xlabel("Step")
        ax.set_ylabel(metric)
        ax.set_yscale("log" if metric != "trace_hessian_covariance" else "symlog")
        ax.legend()
        ax.set_title(metric)

    fig.suptitle(f"Width={width}, Noise={noise}", fontsize=14)
    fig.tight_layout()
    plt.show()

# %%
from dln.utils import get_device

print(get_device())
# %%
from runner import run

result = run(
    "diagonal_teacher",
    overrides={
        "max_steps": 100,
        "evaluate_every": 20,
        "training.batch_size": 5,
        "training.batch_seed": 0,
        "model.seed": 0,
        "data.data_seed": 0,
        "plotting.enabled": False,
    },
    autoplot=False,
)

print(result["step"])
print(result["train_loss"])
# %%
