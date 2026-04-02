"""Hessian spectrum analysis combining offline and online training sweeps.

Separate figure per diagnostic type, each with hidden_dim rows × gamma columns.
Offline (blue) vs Online (red), GD (solid) vs SGD (dashed) overlaid.

Figures:
  - test_loss.png
  - top10_eigenvalues.png
  - eigenvalue_magnitude_dist.png
  - cumulative_spectral_energy.png
  - hessian_trace.png
  - negative_eigenvalue_frac.png
  - full_spectrum/ (one per gamma × hidden_dim × {off,on} × {gd,sgd})

Usage:
    python -m analysis.hessian_spectrum
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from pathlib import Path

from dln.results_io import load_sweep

# ── Config ──────────────────────────────────────────────────────────────

GAMMA_CONFIGS = [(0.75, 5000), (1.0, 8000), (1.5, 26000)]
GAMMA_NAMES = {0.75: "NTK", 1.0: "Mean-Field", 1.5: "Saddle-to-Saddle"}

# Offline = blue family, Online = red family
STYLE = {
    "off_gd":  {"color": "#1f77b4", "ls": "-",  "label": "Offline GD"},
    "off_sgd": {"color": "#6baed6", "ls": "--", "label": "Offline SGD"},
    "on_gd":   {"color": "#d62728", "ls": "-",  "label": "Online GD"},
    "on_sgd":  {"color": "#fc9272", "ls": "--", "label": "Online SGD"},
}

out_dir = Path("figures/hessian_spectrum")

# ── Data loading ────────────────────────────────────────────────────────

off_path = Path("outputs/hessian_spectrum/offline")
on_path = Path("outputs/hessian_spectrum/online")

df_off = load_sweep(off_path) if (off_path / "results.parquet").exists() else None
df_on = load_sweep(on_path) if (on_path / "results.parquet").exists() else None

ref_df = df_off if df_off is not None else df_on
hidden_dims = sorted(ref_df["model.hidden_dim"].unique().to_list())


# ── Helpers ─────────────────────────────────────────────────────────────

def get_row(df, gamma, max_steps, hd, batch_size=None):
    filt = (
        (pl.col("model.gamma") == gamma)
        & (pl.col("max_steps") == max_steps)
        & (pl.col("model.hidden_dim") == hd)
    )
    if batch_size is None:
        filt = filt & pl.col("training.batch_size").is_null()
    else:
        filt = filt & (pl.col("training.batch_size") == batch_size)
    return df.filter(filt)


def load_run(df, gamma, max_steps, hd, batch_size=None):
    if df is None:
        return None
    row = get_row(df, gamma, max_steps, hd, batch_size)
    if len(row) == 0:
        return None
    steps = np.array(row["step"][0])
    loss = np.array(row["test_loss"][0])
    spectra = np.stack(row["hessian_spectrum"][0])  # (n_evals, P)
    return steps, loss, spectra


CACHE_PATH = out_dir / ".cache.npz"
KEYS = ["off_gd", "off_sgd", "on_gd", "on_sgd"]


def _runs_to_flat(runs):
    """Flatten runs dict into arrays for npz serialization."""
    arrays = {}
    for gamma, max_steps in GAMMA_CONFIGS:
        for hd in hidden_dims:
            for key in KEYS:
                run = runs[gamma][hd][key]
                if run is None:
                    continue
                prefix = f"{gamma}_{hd}_{key}"
                arrays[f"{prefix}_steps"] = run[0]
                arrays[f"{prefix}_loss"] = run[1]
                arrays[f"{prefix}_spectra"] = run[2]
    return arrays


def _flat_to_runs(arrays):
    """Reconstruct runs dict from npz arrays."""
    runs = {}
    for gamma, max_steps in GAMMA_CONFIGS:
        runs[gamma] = {}
        for hd in hidden_dims:
            runs[gamma][hd] = {}
            for key in KEYS:
                prefix = f"{gamma}_{hd}_{key}"
                k_steps = f"{prefix}_steps"
                if k_steps in arrays:
                    runs[gamma][hd][key] = (
                        arrays[k_steps],
                        arrays[f"{prefix}_loss"],
                        arrays[f"{prefix}_spectra"],
                    )
                else:
                    runs[gamma][hd][key] = None
    return runs


def load_all_runs():
    """Load all runs, using npz cache if available and fresh."""
    # Check if cache is newer than both parquet files
    if CACHE_PATH.exists():
        cache_mtime = CACHE_PATH.stat().st_mtime
        parquet_files = [
            p / "results.parquet" for p in [off_path, on_path]
            if (p / "results.parquet").exists()
        ]
        if parquet_files and all(cache_mtime > p.stat().st_mtime for p in parquet_files):
            print(f"Loading from cache: {CACHE_PATH}")
            return _flat_to_runs(np.load(CACHE_PATH))

    print("Loading from parquet (first run or data changed)...")
    runs = {}
    for gamma, max_steps in GAMMA_CONFIGS:
        runs[gamma] = {}
        for hd in hidden_dims:
            runs[gamma][hd] = {
                "off_gd":  load_run(df_off, gamma, max_steps, hd, None),
                "off_sgd": load_run(df_off, gamma, max_steps, hd, 1),
                "on_gd":   load_run(df_on, gamma, max_steps, hd, 500),
                "on_sgd":  load_run(df_on, gamma, max_steps, hd, 1),
            }

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_PATH, **_runs_to_flat(runs))
    print(f"Cached to {CACHE_PATH}")
    return runs


def make_grid(title):
    """Create a hidden_dim rows × gamma columns figure."""
    n_rows, n_cols = len(hidden_dims), len(GAMMA_CONFIGS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows),
                              squeeze=False)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    for hd_idx, hd in enumerate(hidden_dims):
        axes[hd_idx, 0].set_ylabel(f"width={hd}", fontsize=10, fontweight="bold")
    for g_idx, (gamma, _) in enumerate(GAMMA_CONFIGS):
        axes[0, g_idx].set_title(f"{GAMMA_NAMES[gamma]} (γ={gamma})",
                                  fontsize=10, fontweight="bold")

    return fig, axes


def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def add_legend_once(ax, g_idx, hd_idx):
    """Add legend only to top-left cell."""
    if g_idx == 0 and hd_idx == 0:
        ax.legend(fontsize=7)


# ── Plot functions ──────────────────────────────────────────────────────

def plot_test_loss(runs):
    fig, axes = make_grid("Test Loss\nOffline (blue) vs Online (red) | GD (solid) vs SGD (dashed)")
    for g_idx, (gamma, _) in enumerate(GAMMA_CONFIGS):
        for hd_idx, hd in enumerate(hidden_dims):
            ax = axes[hd_idx, g_idx]
            for key, sty in STYLE.items():
                run = runs[gamma][hd][key]
                if run:
                    ax.plot(run[0], run[1], color=sty["color"], ls=sty["ls"],
                            linewidth=1, label=sty["label"])
            ax.set_yscale("log")
            ax.set_xlabel("Step", fontsize=8)
            add_legend_once(ax, g_idx, hd_idx)
    save_fig(fig, out_dir / "test_loss.png")


def plot_top10_eigenvalues(runs):
    fig, axes = make_grid("Top-10 Eigenvalue Trajectories\nOffline (blue) vs Online (red) | GD (solid) vs SGD (dashed)")
    for g_idx, (gamma, _) in enumerate(GAMMA_CONFIGS):
        for hd_idx, hd in enumerate(hidden_dims):
            ax = axes[hd_idx, g_idx]
            for key, sty in STYLE.items():
                run = runs[gamma][hd][key]
                if not run:
                    continue
                top_k = run[2][:, -10:][:, ::-1]
                for i in range(top_k.shape[1]):
                    ax.plot(run[0], top_k[:, i], color=sty["color"], ls=sty["ls"],
                            alpha=0.7, linewidth=0.8)
            ax.set_xlabel("Step", fontsize=8)
    save_fig(fig, out_dir / "top10_eigenvalues.png")


def plot_eigenvalue_magnitude(runs):
    """Eigenvalue magnitude distribution — GD only, offline vs online side by side."""
    fig, axes = make_grid("Eigenvalue Magnitude Distribution (GD)\nOffline (blue) vs Online (red)")
    snap_colors_off = ["#1f77b4", "#6baed6", "#9ecae1", "#08519c"]
    snap_colors_on = ["#d62728", "#fc9272", "#fcbba1", "#a50f15"]
    for g_idx, (gamma, _) in enumerate(GAMMA_CONFIGS):
        for hd_idx, hd in enumerate(hidden_dims):
            ax = axes[hd_idx, g_idx]
            for key, snap_colors, regime_lbl in [
                ("off_gd", snap_colors_off, "Off"),
                ("on_gd", snap_colors_on, "On"),
            ]:
                run = runs[gamma][hd][key]
                if not run:
                    continue
                steps, _, spectra = run
                n_evals = spectra.shape[0]
                snap_idx = [0, n_evals // 4, n_evals // 2, -1]
                snap_lbl = ["init", "25%", "50%", "final"]
                for idx, label, c in zip(snap_idx, snap_lbl, snap_colors):
                    eigs = spectra[idx]
                    abs_eigs = np.abs(eigs)
                    abs_eigs = abs_eigs[abs_eigs > 0]
                    ax.hist(np.log10(abs_eigs), bins=60, alpha=0.35, color=c,
                            label=f"{regime_lbl} {label} ({steps[idx]})")
            ax.set_xlabel(r"$\log_{10} |\lambda|$", fontsize=8)
            if g_idx == 0 and hd_idx == 0:
                ax.legend(fontsize=4, ncol=2)
    save_fig(fig, out_dir / "eigenvalue_magnitude_dist.png")


def plot_cumulative_energy(runs):
    fig, axes = make_grid("Cumulative Spectral Energy (GD)\nOffline (blue) vs Online (red)")
    for g_idx, (gamma, _) in enumerate(GAMMA_CONFIGS):
        for hd_idx, hd in enumerate(hidden_dims):
            ax = axes[hd_idx, g_idx]
            for key, color, regime_lbl in [("off_gd", "#1f77b4", "Off"), ("on_gd", "#d62728", "On")]:
                run = runs[gamma][hd][key]
                if not run:
                    continue
                steps, _, spectra = run
                n_evals = spectra.shape[0]
                snap_idx = [0, n_evals // 4, n_evals // 2, -1]
                snap_lbl = ["init", "25%", "50%", "final"]
                alphas = [0.4, 0.6, 0.8, 1.0]
                for idx, label, a in zip(snap_idx, snap_lbl, alphas):
                    eigs_sorted = np.sort(np.abs(spectra[idx]))[::-1]
                    cumsum = np.cumsum(eigs_sorted)
                    total = cumsum[-1] if cumsum[-1] > 0 else 1.0
                    ax.plot(range(1, len(cumsum) + 1), cumsum / total,
                            alpha=a, color=color, linewidth=1,
                            label=f"{regime_lbl} {label} ({steps[idx]})")
            ax.set_xscale("log")
            ax.set_xlabel("Top-k eigenvalues", fontsize=8)
            ax.axhline(0.9, color="gray", linestyle=":", alpha=0.5)
            if g_idx == 0 and hd_idx == 0:
                ax.legend(fontsize=4, ncol=2)
    save_fig(fig, out_dir / "cumulative_spectral_energy.png")


def plot_hessian_trace(runs):
    fig, axes = make_grid("Hessian Trace\nOffline (blue) vs Online (red) | GD (solid) vs SGD (dashed)")
    for g_idx, (gamma, _) in enumerate(GAMMA_CONFIGS):
        for hd_idx, hd in enumerate(hidden_dims):
            ax = axes[hd_idx, g_idx]
            for key, sty in STYLE.items():
                run = runs[gamma][hd][key]
                if run:
                    ax.plot(run[0], run[2].sum(axis=1), color=sty["color"],
                            ls=sty["ls"], linewidth=1, label=sty["label"])
            ax.set_xlabel("Step", fontsize=8)
            add_legend_once(ax, g_idx, hd_idx)
    save_fig(fig, out_dir / "hessian_trace.png")


def plot_negative_frac(runs):
    fig, axes = make_grid("Fraction of Negative Eigenvalues\nOffline (blue) vs Online (red) | GD (solid) vs SGD (dashed)")
    for g_idx, (gamma, _) in enumerate(GAMMA_CONFIGS):
        for hd_idx, hd in enumerate(hidden_dims):
            ax = axes[hd_idx, g_idx]
            for key, sty in STYLE.items():
                run = runs[gamma][hd][key]
                if run:
                    ax.plot(run[0], (run[2] < 0).mean(axis=1), color=sty["color"],
                            ls=sty["ls"], linewidth=1, label=sty["label"])
            ax.set_xlabel("Step", fontsize=8)
            add_legend_once(ax, g_idx, hd_idx)
    save_fig(fig, out_dir / "negative_eigenvalue_frac.png")


def _plot_spectrum_col(ax_loss, ax_off, ax_on, runs_col, col_label):
    """Plot loss + offline/online spectrum into one column of axes."""
    off_color, on_color = "#1f77b4", "#d62728"
    ax_map = {"off": (ax_off, off_color), "on": (ax_on, on_color)}

    for regime in ["off", "on"]:
        run = runs_col[regime]
        if run is None:
            continue
        steps, loss, spectra = run
        ax_spec, color = ax_map[regime]

        # Loss
        label = "Offline" if regime == "off" else "Online"
        ax_loss.plot(steps, loss, color=color, linewidth=1, label=label)

        # Full spectrum
        P = spectra.shape[1]
        cmap = plt.cm.cool
        for i in range(P):
            ax_spec.plot(steps, spectra[:, i], color=cmap(i / max(P - 1, 1)),
                         alpha=0.15, linewidth=0.5)
        for i in range(5):
            ax_spec.plot(steps, spectra[:, -(i + 1)], alpha=0.9, linewidth=1.2,
                         label=rf"$\lambda_{{{P - i}}}$")

    ax_loss.set_yscale("log")
    ax_loss.set_title(col_label, fontsize=10, fontweight="bold")
    ax_loss.legend(fontsize=7)

    for ax_spec, regime_name in [(ax_off, "Offline"), (ax_on, "Online")]:
        ax_spec.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax_spec.set_yscale("symlog", linthresh=1e-7)
        ax_spec.legend(fontsize=7, loc="upper right")
        ax_spec.set_title(regime_name, fontsize=9, loc="left")


def plot_all_full_spectra(runs):
    spectrum_dir = out_dir / "full_spectrum"
    spectrum_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for gamma, _ in GAMMA_CONFIGS:
        gamma_name = GAMMA_NAMES[gamma]
        for hd in hidden_dims:
            gd_runs = {"off": runs[gamma][hd]["off_gd"], "on": runs[gamma][hd]["on_gd"]}
            sgd_runs = {"off": runs[gamma][hd]["off_sgd"], "on": runs[gamma][hd]["on_sgd"]}
            has_any = any(r is not None for r in list(gd_runs.values()) + list(sgd_runs.values()))
            if not has_any:
                continue

            fig, axes = plt.subplots(3, 2, figsize=(16, 12),
                                     height_ratios=[1, 3, 3])
            fig.suptitle(f"Full Hessian Spectrum: {gamma_name} (γ={gamma}), width={hd}",
                         fontsize=12, fontweight="bold", y=1.01)

            # Left column = GD, Right column = SGD
            _plot_spectrum_col(axes[0, 0], axes[1, 0], axes[2, 0], gd_runs, "GD")
            _plot_spectrum_col(axes[0, 1], axes[1, 1], axes[2, 1], sgd_runs, "SGD")

            axes[2, 0].set_xlabel("Step")
            axes[2, 1].set_xlabel("Step")

            fig.tight_layout()
            fig.savefig(spectrum_dir / f"g{gamma}_hd{hd}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            count += 1
    return count


# ── Main ────────────────────────────────────────────────────────────────

out_dir.mkdir(parents=True, exist_ok=True)
runs = load_all_runs()

plot_test_loss(runs)
plot_top10_eigenvalues(runs)
plot_eigenvalue_magnitude(runs)
plot_cumulative_energy(runs)
plot_hessian_trace(runs)
plot_negative_frac(runs)
n_figs = 6

n_spectrum = plot_all_full_spectra(runs)
n_figs += n_spectrum

print(f"All done. {n_figs} figures in {out_dir}/")
