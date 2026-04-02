"""Unified statistics dataclass and factory functions for analysis scripts."""

from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats

from _common import mean_centered_spread


@dataclass
class MetricStats:
    """Per-metric statistics with precomputed CI bounds.

    Superset of all per-script variants. Fields not needed by a given script
    are left as None.
    """

    mean: np.ndarray       # (n_steps,)
    n: int
    ci_lo: np.ndarray      # 95% CI lower
    ci_hi: np.ndarray      # 95% CI upper
    min_vals: np.ndarray | None = None
    max_vals: np.ndarray | None = None
    spread_lo: np.ndarray | None = None  # mean-centered 90% band
    spread_hi: np.ndarray | None = None


def make_stats_log_ci(curves: np.ndarray) -> MetricStats:
    """Log-space 95% CI via delta method. For always-positive metrics."""
    n = len(curves)
    mean = curves.mean(axis=0)

    if n == 1:
        return MetricStats(
            mean=mean, n=1, ci_lo=mean, ci_hi=mean,
            min_vals=None, max_vals=None,
        )

    var = curves.var(axis=0, ddof=1)
    t_val = scipy_stats.t.ppf(0.975, df=n - 1)
    sem = np.sqrt(var / n)

    relative_sem = sem / np.maximum(np.abs(mean), 1e-30)
    ci_factor = np.exp(np.minimum(t_val * relative_sem, 700))

    return MetricStats(
        mean=mean, n=n,
        ci_lo=mean / ci_factor,
        ci_hi=mean * ci_factor,
        min_vals=curves.min(axis=0),
        max_vals=curves.max(axis=0),
    )


def make_stats_linear_ci(curves: np.ndarray) -> MetricStats:
    """Linear 95% CI. For metrics that can be negative (e.g., cosine similarity)."""
    n = len(curves)
    mean = curves.mean(axis=0)

    if n == 1:
        return MetricStats(
            mean=mean, n=1, ci_lo=mean, ci_hi=mean,
            min_vals=None, max_vals=None,
        )

    var = curves.var(axis=0, ddof=1)
    t_val = scipy_stats.t.ppf(0.975, df=n - 1)
    sem = np.sqrt(var / n)

    return MetricStats(
        mean=mean, n=n,
        ci_lo=mean - t_val * sem,
        ci_hi=mean + t_val * sem,
        min_vals=curves.min(axis=0),
        max_vals=curves.max(axis=0),
    )


def make_stats_with_spread(curves: np.ndarray) -> MetricStats:
    """Linear 95% CI plus mean-centered 90% spread bands."""
    n = len(curves)
    mean = curves.mean(axis=0)

    if n == 1:
        return MetricStats(
            mean=mean, n=1,
            ci_lo=None, ci_hi=None,
            min_vals=None, max_vals=None,
            spread_lo=None, spread_hi=None,
        )

    spread_lo, spread_hi = mean_centered_spread(curves, mean)
    t_val = scipy_stats.t.ppf(0.975, df=n - 1)
    sem = np.sqrt(curves.var(axis=0, ddof=1) / n)

    return MetricStats(
        mean=mean, n=n,
        ci_lo=mean - t_val * sem,
        ci_hi=mean + t_val * sem,
        min_vals=curves.min(axis=0),
        max_vals=curves.max(axis=0),
        spread_lo=spread_lo,
        spread_hi=spread_hi,
    )


def wrap_deterministic(arr: np.ndarray) -> MetricStats:
    """Wrap a single deterministic curve as MetricStats with n=1."""
    return MetricStats(
        mean=arr, n=1, ci_lo=arr, ci_hi=arr,
        min_vals=None, max_vals=None,
    )
