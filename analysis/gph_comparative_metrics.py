"""
GPH Comparative Metrics Analysis

Processes both offline (GD vs SGD) and online (B=500 vs mini-batch) regimes.
Figures have gamma values as columns, with separate figures per model seed.

Two figure types per configuration:

  - Distances: loss, relative param/frobenius/layer distances, cosine similarity
  - Layer Distances: per-layer relative weight distances

Usage (run from the project root):

    python analysis/gph_comparative_metrics.py --recompute

Options:
    --offline-input PATH     Offline comparative results dir
    --offline-gd PATH        Offline GD model metrics dir
    --online-input PATH      Online comparative results dir
    --online-baseline PATH   Online baseline (B=500) metrics dir
    --output PATH            Output figures directory
    --recompute              Force recompute statistics (ignore cache)
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from _cache import save_cache as _save_raw, load_cache as _load_raw
from _common import CACHE_DIR, GAMMA_NAMES, BL_KEY_COLS, fmt_seeds, extract_curves
from _parallel import run_pool
from _stats import MetricStats, make_stats_log_ci, make_stats_linear_ci, wrap_deterministic


# =============================================================================
# Configuration
# =============================================================================

OFFLINE_INPUT = Path("outputs/gph_offline_comparative_metrics/comparative")
OFFLINE_GD = Path("outputs/gph_offline_comparative_metrics/gd_metrics")
ONLINE_INPUT = Path("outputs/gph_online_comparative_metrics/comparative")
ONLINE_BASELINE = Path("outputs/gph_online_comparative_metrics/baseline_metrics")
DEFAULT_OUTPUT = Path("figures/gph_comparative_metrics")

N_WORKERS = 10

GROUP_COLS = [
    "model.hidden_dim", "model.gamma", "data.noise_std",
    "model.model_seed", "training_b.batch_size",
]

BATCH_COLORS = {1: "C0", 2: "C2", 5: "C3", 10: "C1", 50: "C4"}


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class CompConfigStats:
    """Unified stats for one (width, gamma, noise, model_seed, batch_size) group.

    Works for both offline (GD ref, n=1) and online (B=500 ref, n>1).
    """

    steps: np.ndarray

    # Loss
    loss_ref: MetricStats      # GD (n=1) or B=500 (n>1)
    loss_sgd: MetricStats

    # Comparative distances
    param_distance: MetricStats
    frobenius_distance: MetricStats
    layer_distances: dict[int, MetricStats]
    cosine_sim: MetricStats | None

    # Reference model norm (for distance normalisation)
    ref_param_norm: np.ndarray | None

    # Reference model metrics (MetricStats; n=1 for offline GD)
    ref_layer_norms: dict[int, MetricStats]
    ref_gram_norms: dict[int, MetricStats]
    ref_balance_diffs: dict[int, MetricStats]
    ref_balance_ratios: dict[int, MetricStats]
    ref_end_to_end: MetricStats | None

    # SGD model metrics
    sgd_layer_norms: dict[int, MetricStats]
    sgd_gram_norms: dict[int, MetricStats]
    sgd_balance_diffs: dict[int, MetricStats]
    sgd_balance_ratios: dict[int, MetricStats]
    sgd_end_to_end: MetricStats | None


# =============================================================================
# Reference Data Loading
# =============================================================================


def _load_gd_ref(gd_dir: Path) -> dict[tuple, dict]:
    """Load deterministic GD model metrics (offline regime).

    Returns dict keyed by (width, gamma, noise, seed).
    Values contain MetricStats-wrapped arrays (n=1).
    """
    path = gd_dir / "results.parquet"
    if not path.exists():
        print(f"  GD metrics not found at {path}")
        return {}

    print(f"  Loading GD metrics from {path}...")
    df = pl.read_parquet(path)

    n_layers = 0
    while f"layer_norm_{n_layers}" in df.columns:
        n_layers += 1
    n_pairs = max(n_layers - 1, 0)

    result = {}
    for row in df.iter_rows(named=True):
        key = (
            row["model.hidden_dim"], row["model.gamma"],
            row["data.noise_std"], row["model.model_seed"],
        )

        layer_norms_raw = {}
        gram_norms_raw = {}
        for i in range(n_layers):
            if f"layer_norm_{i}" in row and row[f"layer_norm_{i}"] is not None:
                layer_norms_raw[i] = np.array(row[f"layer_norm_{i}"])
            if f"gram_norm_{i}" in row and row[f"gram_norm_{i}"] is not None:
                gram_norms_raw[i] = np.array(row[f"gram_norm_{i}"])

        balance_diffs_raw = {}
        balance_ratios_raw = {}
        for i in range(n_pairs):
            col = f"balance_diff_{i}"
            if col in row and row[col] is not None:
                balance_diffs_raw[i] = np.array(row[col])
                if i in gram_norms_raw and (i + 1) in gram_norms_raw:
                    denom = gram_norms_raw[i] + gram_norms_raw[i + 1]
                    balance_ratios_raw[i] = balance_diffs_raw[i] / np.maximum(denom, 1e-30)

        param_norm = (
            np.sqrt(sum(v ** 2 for v in layer_norms_raw.values()))
            if layer_norms_raw else None
        )

        eff_raw = None
        if "end_to_end_weight_norm" in row and row["end_to_end_weight_norm"] is not None:
            eff_raw = np.array(row["end_to_end_weight_norm"])

        result[key] = {
            "layer_norms": {i: wrap_deterministic(v) for i, v in layer_norms_raw.items()},
            "gram_norms": {i: wrap_deterministic(v) for i, v in gram_norms_raw.items()},
            "balance_diffs": {i: wrap_deterministic(v) for i, v in balance_diffs_raw.items()},
            "balance_ratios": {i: wrap_deterministic(v) for i, v in balance_ratios_raw.items()},
            "param_norm": param_norm,
            "end_to_end_weight_norm": wrap_deterministic(eff_raw) if eff_raw is not None else None,
        }

    print(f"  {len(result)} GD configs loaded")
    return result


def _load_online_ref(baseline_dir: Path) -> dict[tuple, dict]:
    """Load stochastic baseline (B=500) model metrics (online regime).

    Returns dict keyed by (width, gamma, noise, seed).
    Values contain MetricStats with CI (n>1).
    """
    path = baseline_dir / "results.parquet"
    if not path.exists():
        print(f"  Baseline metrics not found at {path}")
        return {}

    print(f"  Loading baseline metrics from {path}...")
    df = pl.read_parquet(path)

    n_layers = 0
    while f"layer_norm_{n_layers}" in df.columns:
        n_layers += 1
    n_pairs = max(n_layers - 1, 0)

    groups = df.partition_by(BL_KEY_COLS, as_dict=True)
    result = {}

    for key, gdf in groups.items():
        if len(gdf) == 0:
            continue

        layer_norms = {}
        gram_norms = {}
        for i in range(n_layers):
            if f"layer_norm_{i}" in gdf.columns:
                layer_norms[i] = make_stats_log_ci(extract_curves(gdf, f"layer_norm_{i}"))
            if f"gram_norm_{i}" in gdf.columns:
                gram_norms[i] = make_stats_log_ci(extract_curves(gdf, f"gram_norm_{i}"))

        balance_diffs = {}
        balance_ratios = {}
        for i in range(n_pairs):
            col = f"balance_diff_{i}"
            if col in gdf.columns:
                diff_curves = extract_curves(gdf, col)
                balance_diffs[i] = make_stats_log_ci(diff_curves)
                gcol_l, gcol_r = f"gram_norm_{i}", f"gram_norm_{i + 1}"
                if gcol_l in gdf.columns and gcol_r in gdf.columns:
                    gl = extract_curves(gdf, gcol_l)
                    gr = extract_curves(gdf, gcol_r)
                    balance_ratios[i] = make_stats_linear_ci(
                        diff_curves / np.maximum(gl + gr, 1e-30)
                    )

        param_norm = (
            np.sqrt(sum(v.mean ** 2 for v in layer_norms.values()))
            if layer_norms else None
        )

        eff = None
        if "end_to_end_weight_norm" in gdf.columns:
            eff = make_stats_log_ci(extract_curves(gdf, "end_to_end_weight_norm"))

        result[key] = {
            "layer_norms": layer_norms,
            "gram_norms": gram_norms,
            "balance_diffs": balance_diffs,
            "balance_ratios": balance_ratios,
            "param_norm": param_norm,
            "end_to_end_weight_norm": eff,
        }

    print(f"  {len(result)} baseline configs loaded")
    return result


# =============================================================================
# Stats Computation
# =============================================================================


def _detect_comp_layers(df: pl.DataFrame) -> int:
    i = 0
    while f"layer_distance_{i}" in df.columns:
        i += 1
    return i


def _detect_model_layers(df: pl.DataFrame, suffix: str) -> int:
    i = 0
    while f"layer_norm_{i}_{suffix}" in df.columns:
        i += 1
    return i


def compute_stats(
    comp_dir: Path,
    ref_data: dict[tuple, dict],
    is_online: bool,
) -> dict[tuple, CompConfigStats]:
    """Compute per-config stats. Works for both offline and online regimes."""
    path = comp_dir / "results.parquet"
    print(f"Loading {path}...")
    df = pl.read_parquet(path)
    print(f"  {len(df)} runs loaded")

    n_comp_layers = _detect_comp_layers(df)
    n_b_layers = _detect_model_layers(df, "b")
    has_b_metrics = n_b_layers > 0
    n_b_pairs = max(n_b_layers - 1, 0)
    n_a_layers = _detect_model_layers(df, "a") if is_online else 0

    print(f"  {n_comp_layers} layer distance cols, {n_b_layers} SGD model layers (_b)")
    if n_a_layers > 0:
        print(f"  {n_a_layers} ref model layers (_a)")

    groups = df.partition_by(GROUP_COLS, as_dict=True)
    total = len(groups)
    print(f"  Computing stats for {total} config groups...")

    stats: dict[tuple, CompConfigStats] = {}
    for idx, (key, gdf) in enumerate(groups.items()):
        if len(gdf) == 0:
            continue

        steps = np.array(gdf["step"][0])
        width, gamma, noise, seed, batch_size = key

        # -- Loss --
        loss_ref_curves = extract_curves(gdf, "test_loss_a")
        if is_online:
            loss_ref = make_stats_log_ci(loss_ref_curves)
        else:
            loss_ref = wrap_deterministic(loss_ref_curves.mean(axis=0))
        loss_sgd = make_stats_log_ci(extract_curves(gdf, "test_loss_b"))

        # -- Distances --
        param_dist_curves = extract_curves(gdf, "param_distance")
        param_distance = make_stats_log_ci(param_dist_curves)
        frobenius_distance = make_stats_log_ci(extract_curves(gdf, "frobenius_distance"))

        layer_distances: dict[int, MetricStats] = {}
        for i in range(n_comp_layers):
            layer_distances[i] = make_stats_log_ci(extract_curves(gdf, f"layer_distance_{i}"))

        # -- Ref model metrics (from preloaded ref_data) --
        ref_key = (width, gamma, noise, seed)
        ref = ref_data.get(ref_key, {})
        ref_layer_norms = ref.get("layer_norms", {})
        ref_gram_norms = ref.get("gram_norms", {})
        ref_balance_diffs = ref.get("balance_diffs", {})
        ref_balance_ratios = ref.get("balance_ratios", {})
        ref_param_norm = ref.get("param_norm")
        ref_end_to_end = ref.get("end_to_end_weight_norm")

        # -- SGD model metrics (_b columns) --
        sgd_layer_norms: dict[int, MetricStats] = {}
        sgd_gram_norms: dict[int, MetricStats] = {}
        sgd_balance_diffs: dict[int, MetricStats] = {}
        sgd_balance_ratios: dict[int, MetricStats] = {}
        sgd_end_to_end: MetricStats | None = None
        cosine_sim: MetricStats | None = None

        sgd_ln_curves: dict[int, np.ndarray] = {}
        if has_b_metrics:
            for i in range(n_b_layers):
                sgd_ln_curves[i] = extract_curves(gdf, f"layer_norm_{i}_b")
                sgd_layer_norms[i] = make_stats_log_ci(sgd_ln_curves[i])

            for i in range(n_b_layers):
                sgd_gram_norms[i] = make_stats_log_ci(extract_curves(gdf, f"gram_norm_{i}_b"))

            for i in range(n_b_pairs):
                diff_curves = extract_curves(gdf, f"balance_diff_{i}_b")
                sgd_balance_diffs[i] = make_stats_log_ci(diff_curves)
                gl = extract_curves(gdf, f"gram_norm_{i}_b")
                gr = extract_curves(gdf, f"gram_norm_{i + 1}_b")
                sgd_balance_ratios[i] = make_stats_linear_ci(
                    diff_curves / np.maximum(gl + gr, 1e-30)
                )

            sgd_end_to_end = make_stats_log_ci(
                extract_curves(gdf, "end_to_end_weight_norm_b")
            )

            # -- Cosine similarity (per-run via polarisation identity) --
            if ref_param_norm is not None:
                sgd_pnorm_sq = sum(v ** 2 for v in sgd_ln_curves.values())

                if is_online and n_a_layers > 0:
                    # Per-run ref norms from _a columns
                    ref_pnorm_sq = sum(
                        extract_curves(gdf, f"layer_norm_{i}_a") ** 2
                        for i in range(n_a_layers)
                    )
                    ref_pn = np.sqrt(ref_pnorm_sq)
                else:
                    # Deterministic ref norm (broadcast)
                    ref_pnorm_sq = ref_param_norm ** 2
                    ref_pn = ref_param_norm

                dot = (ref_pnorm_sq + sgd_pnorm_sq - param_dist_curves ** 2) / 2.0
                sgd_pn = np.sqrt(sgd_pnorm_sq)
                denom = ref_pn * np.maximum(sgd_pn, 1e-30)
                cosine_sim = make_stats_linear_ci(dot / denom)

        stats[key] = CompConfigStats(
            steps=steps,
            loss_ref=loss_ref,
            loss_sgd=loss_sgd,
            param_distance=param_distance,
            frobenius_distance=frobenius_distance,
            layer_distances=layer_distances,
            cosine_sim=cosine_sim,
            ref_param_norm=ref_param_norm,
            ref_layer_norms=ref_layer_norms,
            ref_gram_norms=ref_gram_norms,
            ref_balance_diffs=ref_balance_diffs,
            ref_balance_ratios=ref_balance_ratios,
            ref_end_to_end=ref_end_to_end,
            sgd_layer_norms=sgd_layer_norms,
            sgd_gram_norms=sgd_gram_norms,
            sgd_balance_diffs=sgd_balance_diffs,
            sgd_balance_ratios=sgd_balance_ratios,
            sgd_end_to_end=sgd_end_to_end,
        )

        if (idx + 1) % 20 == 0 or idx + 1 == total:
            print(
                f"\r  Stats: {idx + 1}/{total} ({100 * (idx + 1) / total:.0f}%)",
                end="", flush=True,
            )

    print(f"\n  {len(stats)} configurations computed")
    return stats


# =============================================================================
# Caching
# =============================================================================


def _ser_ms(ms: MetricStats | None) -> dict | None:
    return vars(ms) if ms is not None else None


def _de_ms(d: dict | None) -> MetricStats | None:
    return MetricStats(**d) if d is not None else None


def _ser_dict_ms(d: dict[int, MetricStats]) -> dict:
    return {k: vars(v) for k, v in d.items()}


def _de_dict_ms(d: dict) -> dict[int, MetricStats]:
    return {k: MetricStats(**v) for k, v in d.items()}


def _save_cache(stats: dict[tuple, CompConfigStats], path: Path) -> None:
    data = {}
    for key, cs in stats.items():
        data[key] = {
            "steps": cs.steps,
            "loss_ref": _ser_ms(cs.loss_ref),
            "loss_sgd": _ser_ms(cs.loss_sgd),
            "param_distance": _ser_ms(cs.param_distance),
            "frobenius_distance": _ser_ms(cs.frobenius_distance),
            "layer_distances": _ser_dict_ms(cs.layer_distances),
            "cosine_sim": _ser_ms(cs.cosine_sim),
            "ref_param_norm": cs.ref_param_norm,
            "ref_layer_norms": _ser_dict_ms(cs.ref_layer_norms),
            "ref_gram_norms": _ser_dict_ms(cs.ref_gram_norms),
            "ref_balance_diffs": _ser_dict_ms(cs.ref_balance_diffs),
            "ref_balance_ratios": _ser_dict_ms(cs.ref_balance_ratios),
            "ref_end_to_end": _ser_ms(cs.ref_end_to_end),
            "sgd_layer_norms": _ser_dict_ms(cs.sgd_layer_norms),
            "sgd_gram_norms": _ser_dict_ms(cs.sgd_gram_norms),
            "sgd_balance_diffs": _ser_dict_ms(cs.sgd_balance_diffs),
            "sgd_balance_ratios": _ser_dict_ms(cs.sgd_balance_ratios),
            "sgd_end_to_end": _ser_ms(cs.sgd_end_to_end),
        }
    _save_raw(data, path)
    print(f"Cache saved to {path}")


def _load_cache(path: Path) -> dict[tuple, CompConfigStats] | None:
    cache_data = _load_raw(path)
    if cache_data is None:
        return None
    try:
        return {
            key: CompConfigStats(
                steps=d["steps"],
                loss_ref=_de_ms(d["loss_ref"]),
                loss_sgd=_de_ms(d["loss_sgd"]),
                param_distance=_de_ms(d["param_distance"]),
                frobenius_distance=_de_ms(d["frobenius_distance"]),
                layer_distances=_de_dict_ms(d["layer_distances"]),
                cosine_sim=_de_ms(d.get("cosine_sim")),
                ref_param_norm=d.get("ref_param_norm"),
                ref_layer_norms=_de_dict_ms(d.get("ref_layer_norms", {})),
                ref_gram_norms=_de_dict_ms(d.get("ref_gram_norms", {})),
                ref_balance_diffs=_de_dict_ms(d.get("ref_balance_diffs", {})),
                ref_balance_ratios=_de_dict_ms(d.get("ref_balance_ratios", {})),
                ref_end_to_end=_de_ms(d.get("ref_end_to_end")),
                sgd_layer_norms=_de_dict_ms(d.get("sgd_layer_norms", {})),
                sgd_gram_norms=_de_dict_ms(d.get("sgd_gram_norms", {})),
                sgd_balance_diffs=_de_dict_ms(d.get("sgd_balance_diffs", {})),
                sgd_balance_ratios=_de_dict_ms(d.get("sgd_balance_ratios", {})),
                sgd_end_to_end=_de_ms(d.get("sgd_end_to_end")),
            )
            for key, d in cache_data.items()
        }
    except (KeyError, TypeError) as e:
        print(f"Warning: Failed to load cache ({e}), will recompute")
        return None


def get_stats(
    comp_dir: Path,
    ref_data: dict[tuple, dict],
    cache_path: Path,
    is_online: bool,
    force_recompute: bool = False,
) -> dict[tuple, CompConfigStats]:
    if not force_recompute:
        cached = _load_cache(cache_path)
        if cached is not None:
            print(f"Loaded {len(cached)} configurations from cache")
            return cached

    stats = compute_stats(comp_dir, ref_data, is_online)
    _save_cache(stats, cache_path)
    return stats


# =============================================================================
# Plotting Helpers
# =============================================================================


# =============================================================================
# Distances Figure (columns = gamma)
# =============================================================================


def plot_distances(
    stats: dict[tuple, CompConfigStats],
    model_seed: int,
    noise: float,
    width: int,
    batch_sizes: list[int],
    gammas: list[float],
    ref_label: str,
    regime_name: str,
    n_batch_seeds: int | str,
) -> plt.Figure:
    """Distances figure: batch sizes overlaid, gammas as columns.

    Rows: loss, relative param distance, cosine similarity, relative eff-matrix distance.
    """
    has_cosine = any(
        stats.get((width, g, noise, model_seed, b)) is not None
        and stats[(width, g, noise, model_seed, b)].cosine_sim is not None
        for g in gammas for b in batch_sizes
    )

    n_cols = len(gammas)
    n_rows = 3 + int(has_cosine)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 2.5 * n_rows), squeeze=False,
    )

    row_loss = 0
    row_param = 1
    row_cosine = 2 if has_cosine else None
    row_frob = 2 + int(has_cosine)

    cosine_data_min = np.inf
    cosine_data_max = -np.inf

    for col, gamma in enumerate(gammas):
        axes[row_loss, col].set_title(GAMMA_NAMES.get(gamma, str(gamma)), fontsize=10)

        # Plot ref loss once (from first available batch_size)
        for b in batch_sizes:
            k = (width, gamma, noise, model_seed, b)
            if k in stats:
                s = stats[k]
                axes[row_loss, col].plot(
                    s.steps, s.loss_ref.mean,
                    label=ref_label, color="black", linewidth=1.5,
                )
                if s.loss_ref.n > 1:
                    axes[row_loss, col].fill_between(
                        s.steps, s.loss_ref.ci_lo, s.loss_ref.ci_hi,
                        alpha=0.15, color="black",
                    )
                break

        for batch_size in batch_sizes:
            key = (width, gamma, noise, model_seed, batch_size)
            if key not in stats:
                continue
            s = stats[key]
            color = BATCH_COLORS.get(batch_size, "grey")
            lbl = f"B={batch_size}"

            # Loss
            ax = axes[row_loss, col]
            ax.plot(s.steps, s.loss_sgd.mean, color=color, linewidth=1.5, label=lbl)
            ax.fill_between(
                s.steps, s.loss_sgd.ci_lo, s.loss_sgd.ci_hi,
                alpha=0.15, color=color,
            )

            # Relative param distance
            ax = axes[row_param, col]
            if s.ref_param_norm is not None:
                norm = np.maximum(s.ref_param_norm, 1e-30)
                ax.plot(s.steps, s.param_distance.mean / norm,
                        color=color, linewidth=1.5, label=lbl)
                ax.fill_between(
                    s.steps,
                    s.param_distance.ci_lo / norm,
                    s.param_distance.ci_hi / norm,
                    alpha=0.15, color=color,
                )
            else:
                ax.plot(s.steps, s.param_distance.mean,
                        color=color, linewidth=1.5, label=lbl)
                ax.fill_between(
                    s.steps, s.param_distance.ci_lo, s.param_distance.ci_hi,
                    alpha=0.15, color=color,
                )

            # Cosine similarity
            if row_cosine is not None and s.cosine_sim is not None:
                ax = axes[row_cosine, col]
                ax.plot(s.steps, s.cosine_sim.mean,
                        color=color, linewidth=1.5, label=lbl)
                ax.fill_between(
                    s.steps, s.cosine_sim.ci_lo, s.cosine_sim.ci_hi,
                    alpha=0.15, color=color,
                )
                cosine_data_min = min(cosine_data_min, np.nanmin(s.cosine_sim.ci_lo))
                cosine_data_max = max(cosine_data_max, np.nanmax(s.cosine_sim.ci_hi))

            # Relative effective matrix distance
            ax = axes[row_frob, col]
            ref_eff = s.ref_end_to_end.mean if s.ref_end_to_end is not None else None
            if ref_eff is not None:
                norm = np.maximum(ref_eff, 1e-30)
                ax.plot(s.steps, s.frobenius_distance.mean / norm,
                        color=color, linewidth=1.5, label=lbl)
                ax.fill_between(
                    s.steps,
                    s.frobenius_distance.ci_lo / norm,
                    s.frobenius_distance.ci_hi / norm,
                    alpha=0.15, color=color,
                )
            else:
                ax.plot(s.steps, s.frobenius_distance.mean,
                        color=color, linewidth=1.5, label=lbl)
                ax.fill_between(
                    s.steps, s.frobenius_distance.ci_lo, s.frobenius_distance.ci_hi,
                    alpha=0.15, color=color,
                )

        # -- Axis formatting --
        last_col = col == n_cols - 1

        axes[row_loss, col].set_yscale("log")
        if last_col:
            axes[row_loss, col].legend(fontsize=7, loc="best", ncols=2)
        if col == 0:
            axes[row_loss, col].set_ylabel("Test loss", fontsize=10)

        axes[row_param, col].set_yscale("log")
        if last_col:
            axes[row_param, col].legend(fontsize=7, loc="best", ncols=2)
        if col == 0:
            any_key = next(
                (k for k in ((width, gamma, noise, model_seed, b) for b in batch_sizes) if k in stats),
                None,
            )
            has_norm = any_key and stats[any_key].ref_param_norm is not None
            if has_norm:
                axes[row_param, col].set_ylabel(
                    r"$\|\theta_{B} - \theta_{\mathrm{ref}}\|"
                    r" / \|\theta_{\mathrm{ref}}\|$",
                    fontsize=10,
                )
            else:
                axes[row_param, col].set_ylabel(
                    r"$\|\theta_{B} - \theta_{\mathrm{ref}}\|$",
                    fontsize=10,
                )

        if row_cosine is not None:
            axes[row_cosine, col].axhline(
                1.0, color="black", linestyle="--", alpha=0.4, linewidth=0.8,
            )
            if last_col:
                axes[row_cosine, col].legend(fontsize=7, loc="lower right", ncols=2)
            if col == 0:
                axes[row_cosine, col].set_ylabel(
                    r"$\cos(\theta_{\mathrm{ref}}, \theta_{B})$",
                    fontsize=10,
                )

        axes[row_frob, col].set_yscale("log")
        if last_col:
            axes[row_frob, col].legend(fontsize=7, loc="best", ncols=2)
        axes[row_frob, col].set_xlabel("Training step")
        if col == 0:
            any_key = next(
                (k for k in ((width, gamma, noise, model_seed, b) for b in batch_sizes) if k in stats),
                None,
            )
            has_eff = any_key and stats[any_key].ref_end_to_end is not None
            if has_eff:
                axes[row_frob, col].set_ylabel(
                    r"$\|W_{\mathrm{eff}}^{B} - W_{\mathrm{eff}}^{\mathrm{ref}}\|_F"
                    r" / \|W_{\mathrm{eff}}^{\mathrm{ref}}\|_F$",
                    fontsize=10,
                )
            else:
                axes[row_frob, col].set_ylabel(
                    r"$\|W_{\mathrm{eff}}^{B} - W_{\mathrm{eff}}^{\mathrm{ref}}\|_F$",
                    fontsize=10,
                )

    # Synchronize cosine y-axes
    if row_cosine is not None and np.isfinite(cosine_data_min):
        margin = (cosine_data_max - cosine_data_min) * 0.05
        yl = cosine_data_min - margin
        yh = cosine_data_max + margin
        for c in range(n_cols):
            axes[row_cosine, c].set_ylim(yl, yh)
            axes[row_cosine, c].ticklabel_format(useOffset=False)

    fig.tight_layout()
    return fig


# =============================================================================
# Layer Distances Figure (one per batch size, columns = gamma)
# =============================================================================


def plot_layer_distances(
    stats: dict[tuple, CompConfigStats],
    model_seed: int,
    noise: float,
    width: int,
    batch_sizes: list[int],
    gammas: list[float],
    regime_name: str,
    n_batch_seeds: int | str,
) -> plt.Figure | None:
    """Per-layer relative weight distances. One row per batch size, gammas as columns."""
    n_cols = len(gammas)
    has_data = any(
        (width, g, noise, model_seed, b) in stats
        and len(stats[(width, g, noise, model_seed, b)].layer_distances) > 0
        for g in gammas for b in batch_sizes
    )
    if not has_data:
        return None

    n_rows = len(batch_sizes)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 2.5 * n_rows), squeeze=False,
    )

    for col, gamma in enumerate(gammas):
        axes[0, col].set_title(GAMMA_NAMES.get(gamma, str(gamma)), fontsize=10)

    for row, batch_size in enumerate(batch_sizes):
        for col, gamma in enumerate(gammas):
            key = (width, gamma, noise, model_seed, batch_size)
            if key not in stats:
                continue
            s = stats[key]
            ax = axes[row, col]
            for i, ms in sorted(s.layer_distances.items()):
                layer_color = f"C{i}"
                ref_ln = s.ref_layer_norms.get(i)
                if ref_ln is not None:
                    norm = np.maximum(ref_ln.mean, 1e-30)
                    ax.plot(s.steps, ms.mean / norm, color=layer_color,
                            linewidth=1.5, label=f"L{i}")
                    ax.fill_between(
                        s.steps, ms.ci_lo / norm, ms.ci_hi / norm,
                        alpha=0.15, color=layer_color,
                    )
                else:
                    ax.plot(s.steps, ms.mean, color=layer_color,
                            linewidth=1.5, label=f"L{i}")

            ax.set_yscale("log")
            if row == n_rows - 1:
                ax.set_xlabel("Training step")
            if col == 0:
                ax.set_ylabel(
                    r"$\|W_i^{B} - W_i^{\mathrm{ref}}\|_F"
                    r" / \|W_i^{\mathrm{ref}}\|_F$"
                    f"\n(B={batch_size})",
                    fontsize=10,
                )
            if col == n_cols - 1:
                ax.legend(fontsize=7, loc="best", ncols=2)

    fig.tight_layout()
    return fig


# =============================================================================
# Parallel Plot Generation
# =============================================================================

_worker_ctx: dict = {}


def _init_worker(all_stats: dict, output_dir: str) -> None:
    _worker_ctx["stats"] = all_stats
    _worker_ctx["output_dir"] = output_dir


def _run_task(task: tuple) -> None:
    plot_type = task[0]
    stats = _worker_ctx["stats"]
    output_dir = Path(_worker_ctx["output_dir"])

    if plot_type == "distances":
        _, subdir, seed, noise, width, bsizes, gammas, ref_lbl, regime, n_bs, fname = task
        fig = plot_distances(
            stats, seed, noise, width, bsizes, gammas, ref_lbl, regime, n_bs,
        )
    elif plot_type == "layer_distances":
        _, subdir, seed, noise, width, bsizes, gammas, regime, n_bs, fname = task
        fig = plot_layer_distances(
            stats, seed, noise, width, bsizes, gammas, regime, n_bs,
        )
    else:
        return

    if fig is None:
        return
    dest = output_dir / subdir
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest / f"{fname}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_all(
    stats: dict[tuple, CompConfigStats],
    output_dir: Path,
    ref_label: str,
    regime_name: str,
) -> None:
    """Generate all figures for one regime.

    Output structure: output_dir / regime / noise_{noise} / *.png
    """
    regime = regime_name.lower()

    widths = sorted({k[0] for k in stats})
    gammas = sorted({k[1] for k in stats})
    noise_levels = sorted({k[2] for k in stats})
    model_seeds = sorted({k[3] for k in stats})
    batch_sizes = sorted({k[4] for k in stats})

    n_batch_seeds = "?"
    for cs in stats.values():
        n_batch_seeds = cs.loss_sgd.n
        break

    has_layer_dists = any(len(cs.layer_distances) > 0 for cs in stats.values())

    tasks = []
    for model_seed in model_seeds:
        for noise in noise_levels:
            subdir = f"{regime}/noise_{noise}"
            for width in widths:
                tag = f"seed{model_seed}_w{width}"

                # Distances (one per seed/noise/width, all batch sizes overlaid)
                tasks.append((
                    "distances", subdir, model_seed, noise, width,
                    batch_sizes, gammas, ref_label, regime_name, n_batch_seeds,
                    f"distances_{tag}",
                ))

                # Layer distances (all batch sizes on one figure, one row each)
                if has_layer_dists:
                    tasks.append((
                        "layer_distances", subdir, model_seed, noise, width,
                        batch_sizes, gammas, regime_name,
                        n_batch_seeds, f"layer_distances_{tag}",
                    ))

    print(f"Generating {len(tasks)} {regime_name} figures...")

    run_pool(
        _run_task, tasks,
        max_workers=N_WORKERS,
        initializer=_init_worker,
        initargs=(stats, str(output_dir)),
        label=regime_name,
    )

    print(f"{regime_name} plots saved to {output_dir}/")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Plot GD vs SGD metrics for offline and online regimes",
    )
    parser.add_argument(
        "--offline-input", type=Path, default=OFFLINE_INPUT,
        help="Offline comparative results dir",
    )
    parser.add_argument(
        "--offline-gd", type=Path, default=OFFLINE_GD,
        help="Offline GD model metrics dir",
    )
    parser.add_argument(
        "--online-input", type=Path, default=ONLINE_INPUT,
        help="Online comparative results dir",
    )
    parser.add_argument(
        "--online-baseline", type=Path, default=ONLINE_BASELINE,
        help="Online baseline (B=500) metrics dir",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output figures directory",
    )
    parser.add_argument(
        "--recompute", action="store_true",
        help="Force recompute statistics (ignore cache)",
    )
    args = parser.parse_args()

    offline_cache = CACHE_DIR / "gph_combined_offline.pkl"
    online_cache = CACHE_DIR / "gph_combined_online.pkl"

    # --- Offline ---
    comp_path = args.offline_input / "results.parquet"
    if comp_path.exists():
        print("=== Offline Regime ===")
        gd_ref = _load_gd_ref(args.offline_gd)
        offline_stats = get_stats(
            args.offline_input, gd_ref, offline_cache,
            is_online=False, force_recompute=args.recompute,
        )
        generate_all(
            offline_stats, output_dir=args.output,
            ref_label="GD", regime_name="Offline",
        )
    else:
        print(f"Offline input not found at {comp_path}, skipping.")

    # --- Online ---
    comp_path = args.online_input / "results.parquet"
    if comp_path.exists():
        print("\n=== Online Regime ===")
        online_ref = _load_online_ref(args.online_baseline)
        online_stats = get_stats(
            args.online_input, online_ref, online_cache,
            is_online=True, force_recompute=args.recompute,
        )
        generate_all(
            online_stats, output_dir=args.output,
            ref_label="B=500", regime_name="Online",
        )
    else:
        print(f"Online input not found at {comp_path}, skipping.")

    print("\nDone!")


if __name__ == "__main__":
    main()
