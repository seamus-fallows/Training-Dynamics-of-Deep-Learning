from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
from runner import run, load_run

# Config
seeds = list(range(1))
widths = [10, 100]
diagonal_scale = 1
identity_scale = 5
n_workers = 4
use_cpu = False
max_steps_by_width = {10: 70000, 100: 120000}

output_root = Path("outputs/identity_vs_nondegenerate_2")


def run_single(args):
    width, seed, matrix_type = args

    if use_cpu:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if matrix_type == "identity":
        overrides = {
            "data.params.matrix": "identity",
            "data.params.scale": identity_scale,
            "model.hidden_dim": width,
            "model.seed": seed,
            "max_steps": max_steps_by_width[width],
        }
        output_dir = output_root / f"identity_h{width}_s{seed}"
    else:
        overrides = {
            "data.params.scale": diagonal_scale,
            "model.hidden_dim": width,
            "model.seed": seed,
            "max_steps": max_steps_by_width[width],
        }
        output_dir = output_root / f"nondegenerate_h{width}_s{seed}"

    if (output_dir / "history.json").exists():
        return (width, seed, matrix_type, "skipped")

    output_dir.mkdir(parents=True, exist_ok=True)
    run("diagonal_teacher", overrides=overrides, output_dir=output_dir, autoplot=False)
    return (width, seed, matrix_type, "done")


def main():
    output_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    for width in widths:
        for seed in seeds:
            jobs.append((width, seed, "identity"))
            jobs.append((width, seed, "nondegenerate"))

    print(f"Running {len(jobs)} jobs with {n_workers} workers (CPU={use_cpu})...")
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for result in pool.map(run_single, jobs):
            width, seed, matrix_type, status = result
            print(f"  {matrix_type} h={width} s={seed}: {status}")

    # Load and plot
    identity_runs = {width: {} for width in widths}
    nondegenerate_runs = {width: {} for width in widths}

    for width in widths:
        for seed in seeds:
            identity_runs[width][seed] = load_run(
                output_root / f"identity_h{width}_s{seed}"
            )
            nondegenerate_runs[width][seed] = load_run(
                output_root / f"nondegenerate_h{width}_s{seed}"
            )

    fig, axes = plt.subplots(1, 2, figsize=(15, 4))

    for ax, width in zip(axes, widths):
        for s in seeds:
            steps = np.array(identity_runs[width][s]["step"])
            color = f"C{s}"

            ax.plot(
                steps,
                identity_runs[width][s]["train_loss"],
                color=color,
                linestyle="-",
                label=f"Identity s={s}" if width == widths[0] else None,
            )
            ax.plot(
                steps,
                nondegenerate_runs[width][s]["train_loss"],
                color=color,
                linestyle="--",
                label=f"Non-degenerate s={s}" if width == widths[0] else None,
            )

        ax.set_yscale("log")
        ax.set_xlabel("Step")
        ax.set_ylabel("Train Loss")
        ax.set_title(f"hidden_dim={width}")

    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_root / "comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
# %%
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from runner import load_run

# Config
seeds = list(range(1))
widths = [10, 50]
output_root = Path("outputs/identity_vs_nondegenerate_2")

# Load
identity_runs = {width: {} for width in widths}
nondegenerate_runs = {width: {} for width in widths}

for width in widths:
    for seed in seeds:
        identity_runs[width][seed] = load_run(
            output_root / f"identity_h{width}_s{seed}"
        )
        nondegenerate_runs[width][seed] = load_run(
            output_root / f"nondegenerate_h{width}_s{seed}"
        )

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, width in zip(axes, widths):
    for s in seeds:
        steps = np.array(identity_runs[width][s]["step"])

        ax.plot(
            steps,
            identity_runs[width][s]["train_loss"],
            color="C0",
            label="5*Identity" if width == widths[0] else None,
        )
        ax.plot(
            steps,
            nondegenerate_runs[width][s]["train_loss"],
            color="C1",
            label="Non-degenerate - diag(1,...,5)" if width == widths[0] else None,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Train Loss")
    ax.set_title(f"hidden_dim={width}, 4 hidden layers")

axes[0].legend()
fig.tight_layout()
fig.savefig(output_root / "comparison.png", dpi=150, bbox_inches="tight")
plt.show()
# %%
