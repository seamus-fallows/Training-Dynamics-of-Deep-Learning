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
        --output=outputs/my_experiment \\
        --subdir='g{model.gamma}_s{training.batch_seed}' \\
        --skip-existing
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dln.experiment import run_experiment, run_comparative_experiment
from dln.overrides import (
    parse_overrides,
    expand_sweep_params,
    get_output_dir,
    auto_subdir_pattern,
    make_job_subdir,
    check_subdir_uniqueness,
)
from dln.utils import load_config


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
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
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device selection (default: auto)",
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
        help="Output directory (default: outputs/{experiment_name}/{timestamp}_{sweep_params})",
    )
    parser.add_argument(
        "--subdir",
        default=None,
        help="Subdirectory pattern, e.g., 'g{model.gamma}_s{training.batch_seed}'",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip jobs where history.json already exists (requires --output)",
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

    return parser.parse_args()


# =============================================================================
# Job Execution
# =============================================================================


def run_single_job(
    config_name: str,
    config_dir: str,
    overrides: dict[str, Any],
    output_dir: Path,
    show_progress: bool = False,
) -> tuple[bool, str | None]:
    """Run a single experiment job. Returns (success, error_message)."""
    try:
        cfg = load_config(config_name, config_dir, overrides)

        if config_dir == "comparative":
            run_comparative_experiment(
                cfg,
                output_dir=output_dir,
                show_progress=show_progress,
                show_plots=False,
            )
        else:
            run_experiment(
                cfg,
                output_dir=output_dir,
                show_progress=show_progress,
                show_plots=False,
            )
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def run_jobs_sequential(
    config_name: str,
    config_dir: str,
    jobs: list[dict[str, Any]],
    output_dir: Path,
    subdir_pattern: str | None,
    skip_existing: bool,
    fail_fast: bool,
) -> tuple[int, int, int, list[tuple[int, dict, str]]]:
    """Run jobs sequentially. Returns (completed, skipped, failed, errors)."""
    completed = 0
    skipped = 0
    failed = 0
    errors = []

    for i, job in enumerate(jobs):
        subdir = make_job_subdir(i, job, subdir_pattern)
        job_dir = output_dir / subdir

        if skip_existing and (job_dir / "history.json").exists():
            skipped += 1
            continue

        job_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{i + 1}/{len(jobs)}] Running {subdir}...", end=" ", flush=True)

        success, error = run_single_job(config_name, config_dir, job, job_dir)

        if success:
            completed += 1
            print("done")
        else:
            failed += 1
            errors.append((i, job, error))
            print(f"FAILED: {error}")
            if fail_fast:
                break

    return completed, skipped, failed, errors


def run_jobs_parallel(
    config_name: str,
    config_dir: str,
    jobs: list[dict[str, Any]],
    output_dir: Path,
    subdir_pattern: str | None,
    skip_existing: bool,
    fail_fast: bool,
    workers: int,
    device: str,
) -> tuple[int, int, int, list[tuple[int, dict, str]]]:
    """Run jobs in parallel. Returns (completed, skipped, failed, errors)."""
    completed = 0
    skipped = 0
    failed = 0
    errors = []

    jobs_to_run = []
    for i, job in enumerate(jobs):
        subdir = make_job_subdir(i, job, subdir_pattern)
        job_dir = output_dir / subdir

        if skip_existing and (job_dir / "history.json").exists():
            skipped += 1
            continue

        job_dir.mkdir(parents=True, exist_ok=True)
        jobs_to_run.append((i, job, job_dir))

    if not jobs_to_run:
        return completed, skipped, failed, errors

    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    total = len(jobs_to_run)
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for idx, (i, job, job_dir) in enumerate(jobs_to_run):
            show_progress = idx == 0
            future = executor.submit(
                run_single_job, config_name, config_dir, job, job_dir, show_progress
            )
            futures[future] = (i, job)

        for future in as_completed(futures):
            i, job = futures[future]
            success, error = future.result()

            if success:
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
                f"{rate:.1f} jobs/s | ETA: {eta:.0f}s | "
                f"completed: {completed}, failed: {failed}",
                end="",
                flush=True,
            )

    print()
    return completed, skipped, failed, errors


def run_sweep(
    config_name: str,
    comparative: bool,
    jobs: list[dict[str, Any]],
    output_dir: Path,
    subdir_pattern: str | None,
    skip_existing: bool,
    fail_fast: bool,
    workers: int,
    device: str,
) -> None:
    """Run a sweep of jobs."""
    config_dir = "comparative" if comparative else "single"

    print(f"Running {len(jobs)} jobs (workers={workers}, device={device})")
    print(f"Output: {output_dir}")
    print()

    start_time = time.time()

    if workers == 1:
        completed, skipped, failed, errors = run_jobs_sequential(
            config_name,
            config_dir,
            jobs,
            output_dir,
            subdir_pattern,
            skip_existing,
            fail_fast,
        )
    else:
        completed, skipped, failed, errors = run_jobs_parallel(
            config_name,
            config_dir,
            jobs,
            output_dir,
            subdir_pattern,
            skip_existing,
            fail_fast,
            workers,
            device,
        )

    elapsed = time.time() - start_time

    print()
    print("=" * 50)
    print(f"Completed: {completed}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")
    print(f"Time:      {elapsed:.1f}s")

    if errors:
        print()
        print("Errors:")
        for i, job, error in errors[:10]:
            print(f"  Job {i} {job}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    args = parse_args()

    if args.skip_existing and not args.output:
        print("Error: --skip-existing requires --output")
        sys.exit(1)

    overrides = parse_overrides(args.overrides)
    jobs = expand_sweep_params(overrides, args.zip_groups)

    subdir_pattern = args.subdir
    if subdir_pattern is None and len(jobs) > 1:
        subdir_pattern = auto_subdir_pattern(overrides)

    check_subdir_uniqueness(jobs, subdir_pattern)

    config_dir = "comparative" if args.comparative else "single"
    cfg = load_config(args.config_name, config_dir)
    experiment_name = cfg.experiment.name

    output_dir = get_output_dir(experiment_name, overrides, args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_sweep(
        config_name=args.config_name,
        comparative=args.comparative,
        jobs=jobs,
        output_dir=output_dir,
        subdir_pattern=subdir_pattern,
        skip_existing=args.skip_existing,
        fail_fast=args.fail_fast,
        workers=args.workers,
        device=args.device,
    )
