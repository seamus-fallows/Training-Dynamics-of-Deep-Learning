"""Plot SV dynamics: GD (solid) vs SGD (batch_size=1, dashed).

One figure per (gamma_config, partial_product, regime).
Each figure: (3 hidden_dims × 2) rows × 4 columns
  - Row pairs per hidden_dim: loss, singular values
  - Columns: model seeds 0–3

Usage:
    python -m analysis.sv_dynamics
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from pathlib import Path

from dln.results_io import load_sweep

# ── Config ──────────────────────────────────────────────────────────────

L = 3 + 1
in_dim, out_dim = 5, 5
n_plot_cap = 10
partial_products = [(i, j) for i in range(L) for j in range(i, L)]
seeds = [0, 1, 2, 3]

GAMMA_CONFIGS = [(0.75, 5000), (1.0, 8000), (1.5, 26000)]
GAMMA_NAMES = {0.75: "NTK", 1.0: "Mean-Field", 1.5: "Saddle-to-Saddle"}

out_dir = Path("figures/sv_dynamics")

# ── Data loading ────────────────────────────────────────────────────────

df_off = load_sweep(Path("outputs/sv_dynamics/offline"))
df_on = load_sweep(Path("outputs/sv_dynamics/online"))

hidden_dims = sorted(df_off["model.hidden_dim"].unique().to_list())

# Batch sizes: GD = max(batch_size), SGD = min(batch_size)
off_batch_sizes = sorted(
    df_off["training.batch_size"].unique().to_list(),
    key=lambda x: x or float("inf"),
)
on_batch_sizes = sorted(df_on["training.batch_size"].unique().to_list())
off_gd_bs = off_batch_sizes[-1]   # None for offline GD
off_sgd_bs = off_batch_sizes[0]   # 1
on_gd_bs = max(on_batch_sizes)    # 500
on_sgd_bs = min(on_batch_sizes)   # 1


# ── Helpers ─────────────────────────────────────────────────────────────

def get_row(df, gamma, max_steps, hd, seed, batch_size):
    """Get single run from dataframe."""
    filt = (
        (pl.col("model.gamma") == gamma)
        & (pl.col("max_steps") == max_steps)
        & (pl.col("model.hidden_dim") == hd)
        & (pl.col("model.model_seed") == seed)
    )
    if batch_size is None:
        filt = filt & pl.col("training.batch_size").is_null()
    else:
        filt = filt & (pl.col("training.batch_size") == batch_size)
    return df.filter(filt)


def load_run(df, gamma, max_steps, hd, seed, batch_size):
    """Load steps, loss, and SV data for a single run."""
    row = get_row(df, gamma, max_steps, hd, seed, batch_size)
    data = {
        "steps": np.array(row["step"][0]),
        "loss": np.array(row["test_loss"][0]),
    }
    for i, j in partial_products:
        key = f"pp_{i}_{j}_sv"
        data[key] = np.stack(row[key][0])
    return data


# ── Plotting ────────────────────────────────────────────────────────────

regimes = [
    ("offline", "Fixed train set (N=500)", df_off, off_gd_bs, off_sgd_bs),
    ("online",  "Online (infinite data)",  df_on,  on_gd_bs,  on_sgd_bs),
]

n_figs = 0
for gamma, max_steps in GAMMA_CONFIGS:
    gamma_name = GAMMA_NAMES[gamma]

    # Pre-load all data across all hidden dims
    all_runs = {}
    for hd in hidden_dims:
        all_runs[hd] = {}
        for seed in seeds:
            all_runs[hd][seed] = {
                "off_gd": load_run(df_off, gamma, max_steps, hd, seed, off_gd_bs),
                "off_sgd": load_run(df_off, gamma, max_steps, hd, seed, off_sgd_bs),
                "on_gd": load_run(df_on, gamma, max_steps, hd, seed, on_gd_bs),
                "on_sgd": load_run(df_on, gamma, max_steps, hd, seed, on_sgd_bs),
            }

    for regime_key, regime_label, _, _, _ in regimes:
        if regime_key == "offline":
            gd_key, sgd_key = "off_gd", "off_sgd"
        else:
            gd_key, sgd_key = "on_gd", "on_sgd"

        regime_dir = out_dir / regime_key / f"g{gamma}"
        regime_dir.mkdir(parents=True, exist_ok=True)

        for i, j in partial_products:
            pp_key = f"pp_{i}_{j}_sv"
            product_label = f"W_{j}" if i == j else f"W_{j} ··· W_{i}"

            n_rows = len(hidden_dims) * 2
            fig, axes = plt.subplots(n_rows, len(seeds), figsize=(4.5 * len(seeds), 3.2 * n_rows))

            for hd_idx, hd in enumerate(hidden_dims):
                row_loss = hd_idx * 2
                row_sv = hd_idx * 2 + 1
                n_svs = all_runs[hd][seeds[0]][gd_key][pp_key].shape[1]
                n_plot = min(n_plot_cap, n_svs)

                for col, seed in enumerate(seeds):
                    r = all_runs[hd][seed]

                    # Loss row
                    ax = axes[row_loss, col]
                    ax.plot(r[gd_key]["steps"], r[gd_key]["loss"], label="GD")
                    ax.plot(r[sgd_key]["steps"], r[sgd_key]["loss"], ls="--", label="SGD")
                    ax.set_yscale("log")
                    if hd_idx == 0:
                        ax.set_title(f"Model Seed {seed}")
                    if col == 0:
                        ax.set_ylabel(f"hidden dim = {hd}\nTest Loss")
                    if col == len(seeds) - 1:
                        ax.legend(fontsize=7)

                    # SV row
                    ax = axes[row_sv, col]
                    for k in range(n_plot):
                        line, = ax.plot(r[gd_key]["steps"], r[gd_key][pp_key][:, k], label=f"\u03c3_{k}")
                        ax.plot(r[sgd_key]["steps"], r[sgd_key][pp_key][:, k], ls="--", color=line.get_color())
                    if col == len(seeds) - 1:
                        ax.legend(fontsize=6, ncol=2, loc="center right")
                    if col == 0:
                        ax.set_ylabel("Singular Value")
                    if hd_idx == len(hidden_dims) - 1:
                        ax.set_xlabel("Step")

            fig.suptitle(
                f"Singular values of {product_label}: GD (solid) vs SGD (batch size = 1, dashed)\n"
                f"{regime_label} | {gamma_name} initialisation (\u03b3 = {gamma})",
                fontsize=13, fontweight="bold",
            )
            fig.tight_layout(rect=[0, 0, 1, 0.985])
            path = regime_dir / f"pp_{i}_{j}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            n_figs += 1

    print(f"\u03b3={gamma}: saved {2 * len(partial_products)} figures (offline + online)")

print(f"\nAll done. {n_figs} figures in {out_dir}/")
