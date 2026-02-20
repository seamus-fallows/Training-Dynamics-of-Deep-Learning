"""
Lightweight experiment runner with parallel sweep support.

Usage:
    # Single run
    python sweep.py -cn=gph model.gamma=0.75

    # Parallel sweep
    python sweep.py -cn=gph training.batch_seed=0..100 --workers=40

    # Covarying params
    python sweep.py -cn=gph model.gamma=0.75,1.0,1.5 max_steps=5000,10000,27000 --zip=model.gamma,max_steps

    # Full example
    python sweep.py -cn=gph \\
        model.gamma=0.75,1.0,1.5 \\
        max_steps=5000,10000,27000 \\
        training.batch_seed=0..100 \\
        --zip=model.gamma,max_steps \\
        --workers=40 \\
        --device=cuda \\
        --output=outputs/my_experiment

    # Re-run specific jobs
    python sweep.py -cn=gph training.batch_seed=0..100 --rerun training.batch_seed=42..50
"""

import torch as t
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any
import traceback

from omegaconf import OmegaConf

from dln.experiment import run_experiment, run_comparative_experiment
from dln.utils import load_base_config, resolve_config, save_sweep_config
from dln.overrides import (
    parse_overrides,
    split_overrides,
    expand_sweep_params,
    get_output_dir,
)
from dln.results_io import SweepWriter, NullWriter


def _fmt_time(seconds):
    h, remainder = divmod(int(seconds), 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run experiments with optional parallel sweeps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-cn",
        "--config-name",
        required=True,
        help="Name of config file (without .yaml)",
    )
    parser.add_argument(
        "--comparative",
        action="store_true",
        help="Run comparative experiment",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = sequential)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default="cuda",
        help="Device selection (default: cuda)",
    )
    parser.add_argument(
        "--zip",
        action="append",
        dest="zip_groups",
        metavar="PARAMS",
        help="Comma-separated param names to zip together (can use multiple times)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: outputs/{experiment_name}/{timestamp})",
    )
    parser.add_argument(
        "--rerun",
        nargs="*",
        metavar="OVERRIDE",
        help="Force re-run of matching jobs (same override syntax as positional args)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first error (default: continue and report at end)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in format param=value or param=val1,val2,val3",
    )
    parser.add_argument(
        "--no-save",
        action="store_false",
        dest="save_results",
        help="Run experiments without saving results (for testing)",
    )

    return parser.parse_args()


# =============================================================================
# Worker Pool
# =============================================================================

_worker_state = {}


def _worker_init(resolved_base, config_dir, device):
    _worker_state.update(
        resolved_base=resolved_base,
        config_dir=config_dir,
        device=device,
    )


def _worker_run_job(job_overrides):
    return run_single_job(
        _worker_state["resolved_base"],
        _worker_state["config_dir"],
        job_overrides,
        _worker_state["device"],
    )


# =============================================================================
# Job Execution
# =============================================================================


def run_single_job(
    resolved_base: dict,
    config_dir: str,
    job_overrides: dict[str, Any],
    device: str,
) -> tuple[bool, dict | None, str | None]:
    """Returns (success, history, error_message)."""
    try:
        cfg = resolve_config(resolved_base, config_dir, job_overrides)

        if config_dir == "comparative":
            result = run_comparative_experiment(cfg, device=device)
        else:
            result = run_experiment(cfg, device=device)

        return True, result.history, None
    except Exception:
        return False, None, traceback.format_exc()


def run_jobs_sequential(
    resolved_base: dict,
    config_dir: str,
    jobs: list[dict[str, Any]],
    writer: SweepWriter | NullWriter,
    fail_fast: bool,
    device: str,
) -> tuple[int, int, list[tuple[int, dict, str]]]:
    """Returns (completed, failed, errors)."""
    completed = 0
    failed = 0
    errors = []

    for i, job in enumerate(jobs):
        if len(jobs) > 1:
            print(f"[{i + 1}/{len(jobs)}]")

        success, history, error = run_single_job(
            resolved_base, config_dir, job, device=device
        )

        if success:
            writer.add(job, history)
            completed += 1
        else:
            failed += 1
            errors.append((i, job, error))
            print(f"FAILED: {error}")
            if fail_fast:
                break

    return completed, failed, errors


def run_jobs_parallel(
    resolved_base: dict,
    config_dir: str,
    jobs: list[dict[str, Any]],
    writer: SweepWriter | NullWriter,
    fail_fast: bool,
    num_workers: int,
    device: str,
) -> tuple[int, int, list[tuple[int, dict, str]]]:
    """Returns (completed, failed, errors)."""
    completed = 0
    failed = 0
    errors = []

    total = len(jobs)
    start_time = time.time()

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_init,
        initargs=(resolved_base, config_dir, device),
    ) as executor:
        futures = {}
        for i, job in enumerate(jobs):
            future = executor.submit(_worker_run_job, job)
            futures[future] = (i, job)

        for future in as_completed(futures):
            i, job = futures[future]
            success, history, error = future.result()

            if success:
                writer.add(job, history)
                completed += 1
            else:
                failed += 1
                errors.append((i, job, error))
                if fail_fast:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

            done = completed + failed
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(
                f"\rProgress: {done}/{total} ({100 * done / total:.1f}%) | "
                f"{rate:.1f} jobs/s | Elapsed: {_fmt_time(elapsed)} | "
                f"ETA: {_fmt_time(eta)} | failed: {failed}",
                end="",
                flush=True,
            )

    print()
    return completed, failed, errors


def run_sweep(
    resolved_base: dict,
    config_dir: str,
    jobs: list[dict[str, Any]],
    writer: SweepWriter | NullWriter,
    fail_fast: bool,
    workers: int,
    device: str,
    skipped: int = 0,
) -> None:
    start_time = time.time()

    print(f"Running {len(jobs)} jobs (workers={workers}, device={device})")
    if skipped:
        print(f"Skipping {skipped} already-completed jobs")
    print()

    if not jobs:
        completed, failed, errors = 0, 0, []
    elif workers == 1:
        completed, failed, errors = run_jobs_sequential(
            resolved_base, config_dir, jobs, writer, fail_fast, device,
        )
    else:
        completed, failed, errors = run_jobs_parallel(
            resolved_base, config_dir, jobs, writer, fail_fast, workers, device,
        )

    writer.finalize()

    elapsed = time.time() - start_time

    print()
    print("=" * 50)
    print(f"Completed: {completed}")
    if skipped:
        print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")
    print(f"Time:      {_fmt_time(elapsed)}")

    if errors:
        print()
        print("Errors:")
        for i, job, error in errors[:10]:
            print(f"  Job {i} {job}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


# =============================================================================
# Resume / Rerun Helpers
# =============================================================================


def _build_rerun_set(
    rerun_args: list[str], param_keys: list[str], completed: set[tuple],
) -> set[tuple]:
    """Supports partial-key matching: ``--rerun model.gamma=0.75`` invalidates all
    completed runs where model.gamma=0.75, regardless of other param values.
    """
    rerun_overrides = parse_overrides(rerun_args)
    rerun_fixed, rerun_sweep_params = split_overrides(rerun_overrides)

    # Warn if rerun keys don't overlap with sweep param keys
    rerun_keys = set(rerun_sweep_params.keys()) | set(rerun_fixed.keys())
    if rerun_keys and not rerun_keys & set(param_keys):
        print(
            f"Warning: --rerun keys {rerun_keys} don't overlap with sweep "
            f"parameters {set(param_keys)}. No jobs will be re-run.",
            file=sys.stderr,
        )

    # Treat fixed values as single-element sweeps for expansion
    for k, v in rerun_fixed.items():
        rerun_sweep_params[k] = [v]

    rerun_jobs = expand_sweep_params(rerun_sweep_params)

    # Build index mapping: which positions in param_keys are constrained by rerun
    match_keys = [k for k in param_keys if k in rerun_keys]
    if not match_keys:
        return set()

    match_indices = [param_keys.index(k) for k in match_keys]

    # Collect the set of partial tuples (only constrained positions)
    rerun_partials = {
        tuple(job[k] for k in match_keys)
        for job in rerun_jobs
    }

    # Match completed tuples by checking only the constrained positions
    invalidated = set()
    for ct in completed:
        partial = tuple(ct[i] for i in match_indices)
        if partial in rerun_partials:
            invalidated.add(ct)
    return invalidated


def _filter_jobs(
    jobs: list[dict[str, Any]],
    param_keys: list[str],
    completed: set[tuple],
) -> tuple[list[dict[str, Any]], int]:
    """Returns (jobs_to_run, skipped_count)."""
    if not completed:
        return jobs, 0

    jobs_to_run = []
    skipped = 0
    for job in jobs:
        key = tuple(job.get(k) for k in param_keys)
        if key in completed:
            skipped += 1
        else:
            jobs_to_run.append(job)
    return jobs_to_run, skipped


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    # Single-threaded BLAS: our matrix ops are too small (e.g. 100x100) for
    # thread parallelism to outweigh wake/sync overhead. Also prevents thread
    # contention when running parallel sweeps (N workers × M BLAS threads).
    # If running single jobs with hidden_dim >> 500, removing these may help.
    t.set_num_threads(1)
    t.set_num_interop_threads(1)

    args = parse_args()

    overrides = parse_overrides(args.overrides)
    fixed_overrides, sweep_overrides = split_overrides(overrides)

    jobs = expand_sweep_params(sweep_overrides, args.zip_groups)
    param_keys = list(sweep_overrides.keys())

    config_dir = "comparative" if args.comparative else "single"
    base_config = load_base_config(args.config_name, config_dir)

    # Resolve base config with fixed overrides baked in
    effective_cfg = resolve_config(base_config, config_dir, fixed_overrides)
    experiment_name = effective_cfg.experiment.name
    resolved_base = OmegaConf.to_container(effective_cfg, resolve=True)

    if args.save_results:
        output_dir = get_output_dir(experiment_name, args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_sweep_config(resolved_base, output_dir)
        writer = SweepWriter(output_dir, param_keys)
    else:
        writer = NullWriter()

    # Crash recovery: consolidate leftover parts from a previous interrupted run
    writer.consolidate_parts()

    # Build completed set and filter jobs
    completed = writer.get_completed_params()

    if args.rerun:
        rerun_set = _build_rerun_set(args.rerun, param_keys, completed)
        completed -= rerun_set

    jobs_to_run, skipped = _filter_jobs(jobs, param_keys, completed)

    run_sweep(
        resolved_base=resolved_base,
        config_dir=config_dir,
        jobs=jobs_to_run,
        writer=writer,
        fail_fast=args.fail_fast,
        workers=args.workers,
        device=args.device,
        skipped=skipped,
    )
