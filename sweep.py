"""
Lightweight experiment runner with parallel sweep support.

Usage:
    # Single run
    python sweep.py -cn=gph model.gamma=0.75

    # Parallel sweep
    python sweep.py -cn=gph training.batch_seed=range(0,100) --workers=40

    # Covarying params
    python sweep.py -cn=gph model.gamma=0.75,1.0,1.5 max_steps=5000,10000,27000 --zip=model.gamma,max_steps

    # Full example
    python sweep.py -cn=gph \\
        model.gamma=0.75,1.0,1.5 \\
        max_steps=5000,10000,27000 \\
        training.batch_seed=range(0,100) \\
        --zip=model.gamma,max_steps \\
        --workers=40 \\
        --device=cuda \\
        --output=outputs/my_experiment \\
        --subdir='g{model.gamma}_s{training.batch_seed}' \\
        --skip-existing
"""

import argparse
import re
from typing import Any
from itertools import product
from pathlib import Path
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def make_output_dir(config_name: str, output_arg: str | None) -> Path:
    """Create and return the output directory for a sweep."""
    if output_arg:
        output_dir = Path(output_arg)
    else:
        output_dir = Path("outputs/sweeps") / config_name

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def format_subdir(pattern: str, overrides: dict[str, Any]) -> str:
    """
    Format a subdir pattern with override values.

    Pattern like 'g{model.gamma}_s{training.batch_seed}' becomes 'g0.75_s0'
    """
    result = pattern
    for key, value in overrides.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, str(value))
    return result


def make_job_subdir(
    job_index: int,
    job_overrides: dict[str, Any],
    subdir_pattern: str | None,
) -> str:
    """Generate subdirectory name for a job."""
    if subdir_pattern:
        return format_subdir(subdir_pattern, job_overrides)
    else:
        return str(job_index)


def check_subdir_uniqueness(
    jobs: list[dict[str, Any]], subdir_pattern: str | None
) -> None:
    """Verify all jobs produce unique subdirectory names."""
    if subdir_pattern is None:
        return  # Index-based subdirs are always unique

    subdirs = {}
    for i, job in enumerate(jobs):
        subdir = format_subdir(subdir_pattern, job)
        if subdir in subdirs:
            prev_job = subdirs[subdir]
            raise ValueError(
                f"Duplicate subdir '{subdir}' for jobs:\n"
                f"  Job {prev_job[0]}: {prev_job[1]}\n"
                f"  Job {i}: {job}\n"
                f"Add more parameters to --subdir pattern to make unique."
            )
        subdirs[subdir] = (i, job)


def parse_value(value_str: str) -> Any | list[Any]:
    """
    Parse a CLI value string into Python value(s).

    Handles:
        - range(start, stop) or range(start, stop, step)
        - Range shorthand: start..stop or start..stop..step
        - Comma-separated values: 1,2,3 or 0.5,1.0,1.5
        - null/None -> None
        - true/false -> bool
        - Numbers (int or float)
        - Strings
    """
    value_str = value_str.strip()

    # Check for range(...)
    range_match = re.match(r"range\((\d+),\s*(\d+)(?:,\s*(\d+))?\)", value_str)
    if range_match:
        start = int(range_match.group(1))
        stop = int(range_match.group(2))
        step = int(range_match.group(3)) if range_match.group(3) else 1
        return list(range(start, stop, step))

    # Check for range shorthand: start..stop or start..stop..step
    range_shorthand = re.match(r"(\d+)\.\.(\d+)(?:\.\.(\d+))?$", value_str)
    if range_shorthand:
        start = int(range_shorthand.group(1))
        stop = int(range_shorthand.group(2))
        step = int(range_shorthand.group(3)) if range_shorthand.group(3) else 1
        return list(range(start, stop, step))

    # Check for comma-separated values
    if "," in value_str:
        return [_parse_single_value(v.strip()) for v in value_str.split(",")]

    return _parse_single_value(value_str)


def _parse_single_value(value_str: str) -> Any:
    """Parse a single value string into Python type."""
    if value_str.lower() in ("null", "none"):
        return None
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False

    # Try int
    try:
        return int(value_str)
    except ValueError:
        pass

    # Try float
    try:
        return float(value_str)
    except ValueError:
        pass

    return value_str


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run experiments with optional parallel sweeps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Config selection
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

    # Parallelization
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

    # Sweep control
    parser.add_argument(
        "--zip",
        action="append",
        dest="zip_groups",
        metavar="PARAMS",
        help="Comma-separated param names to zip together (can use multiple times)",
    )

    # Output control
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: outputs/sweeps/{config_name}_{timestamp})",
    )
    parser.add_argument(
        "--subdir",
        type=str,
        default=None,
        help="Subdirectory pattern, e.g., 'g{model.gamma}_s{training.batch_seed}'",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip jobs where history.json already exists",
    )

    # Error handling
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first error (default: continue and report at end)",
    )

    # Collect overrides (param=value arguments)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in format param=value or param=val1,val2,val3",
    )

    return parser.parse_args()


def parse_overrides(override_args: list[str]) -> dict[str, Any]:
    """
    Parse override arguments into a dictionary.

    Returns dict where values are either:
        - Single values (for regular overrides)
        - Lists (for sweep parameters)
    """
    overrides = {}
    for arg in override_args:
        if "=" not in arg:
            raise ValueError(f"Invalid override format: {arg!r} (expected param=value)")
        key, value_str = arg.split("=", 1)
        overrides[key] = parse_value(value_str)
    return overrides


def expand_sweep_params(
    overrides: dict[str, Any],
    zip_groups: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Expand sweep parameters into a list of job configurations.

    Args:
        overrides: Dict where values may be single values or lists (sweep params)
        zip_groups: List of comma-separated param names to zip together

    Returns:
        List of override dicts, one per job
    """
    # Parse zip groups
    zipped_params = set()
    zip_group_lists = []
    if zip_groups:
        for group in zip_groups:
            params = [p.strip() for p in group.split(",")]
            zip_group_lists.append(params)
            zipped_params.update(params)

    # Separate into: fixed, zipped, and cartesian sweep params
    fixed = {}
    cartesian_sweep = {}

    for key, value in overrides.items():
        if not isinstance(value, list):
            fixed[key] = value
        elif key in zipped_params:
            pass  # Handled separately
        else:
            cartesian_sweep[key] = value

    # Build zipped value lists
    zipped_value_lists = []
    for params in zip_group_lists:
        # Get values for each param in the group
        values_per_param = []
        for p in params:
            if p not in overrides:
                raise ValueError(f"Zip param {p!r} not found in overrides")
            v = overrides[p]
            if not isinstance(v, list):
                raise ValueError(f"Zip param {p!r} must have multiple values")
            values_per_param.append(v)

        # Check all same length
        lengths = [len(v) for v in values_per_param]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"Zip group {params} has mismatched lengths: {dict(zip(params, lengths))}"
            )

        # Zip them together: [(p1_v1, p2_v1), (p1_v2, p2_v2), ...]
        zipped = list(zip(*values_per_param))
        zipped_value_lists.append((params, zipped))

    # Build all combinations
    jobs = []

    # Get cartesian product components
    cartesian_keys = list(cartesian_sweep.keys())
    cartesian_values = [cartesian_sweep[k] for k in cartesian_keys]

    # Get zipped components as single "dimensions" for product
    zipped_keys = [params for params, _ in zipped_value_lists]
    zipped_values = [values for _, values in zipped_value_lists]

    # If no sweep params at all, return single job with fixed values
    if not cartesian_values and not zipped_values:
        return [fixed.copy()]

    # Combine: cartesian product of (cartesian params) x (each zip group)
    all_iterables = cartesian_values + zipped_values

    for combo in product(*all_iterables) if all_iterables else [()]:
        job = fixed.copy()

        # Add cartesian values
        idx = 0
        for key in cartesian_keys:
            job[key] = combo[idx]
            idx += 1

        # Add zipped values
        for params in zipped_keys:
            zipped_tuple = combo[idx]
            for param, value in zip(params, zipped_tuple):
                job[param] = value
            idx += 1

        jobs.append(job)

    return jobs


def run_single_job(
    config_name: str,
    config_dir: str,
    overrides: dict[str, Any],
    output_dir: Path,
    show_progress: bool = False,
) -> tuple[bool, str | None]:
    """
    Run a single experiment job.

    Returns:
        (success, error_message)
    """
    from dln.utils import load_config
    from run import run_experiment
    from run_comparative import run_comparative_experiment

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
        return False, str(e)


def worker_init(device: str) -> None:
    """Initialize worker process."""
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def run_jobs_sequential(
    config_name: str,
    config_dir: str,
    jobs: list[dict[str, Any]],
    output_dir: Path,
    subdir_pattern: str | None,
    skip_existing: bool,
    fail_fast: bool,
) -> tuple[int, int, int, list[tuple[int, dict, str]]]:
    """
    Run jobs sequentially.

    Returns:
        (completed, skipped, failed, error_details)
    """
    completed = 0
    skipped = 0
    failed = 0
    errors = []

    for i, job in enumerate(jobs):
        subdir = make_job_subdir(i, job, subdir_pattern)
        job_dir = output_dir / subdir

        # Skip if exists
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
    """
    Run jobs in parallel using ProcessPoolExecutor.

    Returns:
        (completed, skipped, failed, error_details)
    """
    completed = 0
    skipped = 0
    failed = 0
    errors = []

    # Build list of jobs to run (respecting skip_existing)
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

    # Set device for workers
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    total = len(jobs_to_run)
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for idx, (i, job, job_dir) in enumerate(jobs_to_run):
            # Only first job shows progress bar
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

            # Progress update
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

    print()  # Newline after progress
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

    # Summary
    print()
    print(f"{'=' * 50}")
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


if __name__ == "__main__":
    args = parse_args()
    overrides = parse_overrides(args.overrides)
    jobs = expand_sweep_params(overrides, args.zip_groups)

    check_subdir_uniqueness(jobs, args.subdir)
    output_dir = make_output_dir(args.config_name, args.output)

    run_sweep(
        config_name=args.config_name,
        comparative=args.comparative,
        jobs=jobs,
        output_dir=output_dir,
        subdir_pattern=args.subdir,
        skip_existing=args.skip_existing,
        fail_fast=args.fail_fast,
        workers=args.workers,
        device=args.device,
    )
