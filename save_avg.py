"""
Compute and save averaged SGD results, then delete individual runs.
"""

# %%
import json
import shutil
import numpy as np
from pathlib import Path

from runner import load_run
from plotting import compute_ci

# %%
BASE_PATH = Path("outputs/gph_w100")

N_SEEDS = 100
BATCH_SIZE = 5

WIDTHS = [10, 100]
GAMMAS = [0.75, 1.0, 1.5]
NOISE_LEVELS = [0.0, 0.2]

DELETE_INDIVIDUAL = True  # Set False to keep individual runs


# %%
def get_path(
    width: int, gamma: float, batch: int | None, seed: int, online: bool, noise: float
) -> Path:
    b_str = "None" if batch is None else str(batch)
    return BASE_PATH / f"h{width}_g{gamma}_b{b_str}_s{seed}_online{online}_noise{noise}"


def get_avg_path(width: int, gamma: float, online: bool, noise: float) -> Path:
    return (
        BASE_PATH / f"h{width}_g{gamma}_b{BATCH_SIZE}_avg_online{online}_noise{noise}"
    )


def get_sgd_paths(width: int, gamma: float, online: bool, noise: float) -> list[Path]:
    paths = []
    for seed in range(N_SEEDS):
        p = get_path(width, gamma, BATCH_SIZE, seed, online, noise)
        if p.exists() and (p / "history.json").exists():
            paths.append(p)
    return paths


def average_runs(paths: list[Path]) -> dict:
    """Load runs and compute mean/CI for all metrics."""
    runs = [load_run(p) for p in paths]

    steps = runs[0]["step"]
    result = {"step": steps, "n_runs": len(runs)}

    for metric in runs[0].history.keys():
        if metric == "step":
            continue
        curves = [np.array(r[metric]) for r in runs]
        mean, lower, upper = compute_ci(curves)
        result[f"{metric}_mean"] = mean.tolist()
        result[f"{metric}_lower"] = lower.tolist()
        result[f"{metric}_upper"] = upper.tolist()

    return result


# %%
for width in WIDTHS:
    for gamma in GAMMAS:
        for online in [False, True]:
            for noise in NOISE_LEVELS:
                paths = get_sgd_paths(width, gamma, online, noise)

                if not paths:
                    print(
                        f"No data: w={width}, g={gamma}, online={online}, noise={noise}"
                    )
                    continue

                print(
                    f"Processing: w={width}, g={gamma}, online={online}, noise={noise} ({len(paths)} runs)"
                )

                # Compute average
                avg = average_runs(paths)

                # Save to single file
                avg_path = get_avg_path(width, gamma, online, noise)
                avg_path.mkdir(exist_ok=True)
                with open(avg_path / "history.json", "w") as f:
                    json.dump(avg, f)

                # Delete individual runs
                if DELETE_INDIVIDUAL:
                    for p in paths:
                        shutil.rmtree(p)
                    print(f"  Saved average, deleted {len(paths)} individual runs")
                else:
                    print(f"  Saved average")

print("\nDone!")
# %%
