"""
Override parsing and sweep expansion utilities.
"""

import re
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any


# =============================================================================
# Value Parsing
# =============================================================================


def _parse_single_value(value_str: str) -> Any:
    """Parse a single value string into Python type."""
    if value_str.lower() in ("null", "none"):
        return None
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False

    try:
        return int(value_str)
    except ValueError:
        pass

    try:
        return float(value_str)
    except ValueError:
        pass

    return value_str


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

    range_match = re.match(r"range\((\d+),\s*(\d+)(?:,\s*(\d+))?\)", value_str)
    if range_match:
        start = int(range_match.group(1))
        stop = int(range_match.group(2))
        step = int(range_match.group(3)) if range_match.group(3) else 1
        return list(range(start, stop, step))

    range_shorthand = re.match(r"(\d+)\.\.(\d+)(?:\.\.(\d+))?$", value_str)
    if range_shorthand:
        start = int(range_shorthand.group(1))
        stop = int(range_shorthand.group(2))
        step = int(range_shorthand.group(3)) if range_shorthand.group(3) else 1
        return list(range(start, stop, step))

    if "," in value_str:
        return [_parse_single_value(v.strip()) for v in value_str.split(",")]

    return _parse_single_value(value_str)


def parse_overrides(override_args: list[str]) -> dict[str, Any]:
    """Parse override arguments into a dictionary."""
    overrides = {}
    for arg in override_args:
        if "=" not in arg:
            raise ValueError(f"Invalid override format: {arg!r} (expected param=value)")
        key, value_str = arg.split("=", 1)
        overrides[key] = parse_value(value_str)
    return overrides


# =============================================================================
# Job Expansion
# =============================================================================


def _validate_zip_group(params: list[str], overrides: dict) -> None:
    """Validate that a zip group has matching lengths and all params exist."""
    for p in params:
        if p not in overrides:
            raise ValueError(f"Zip param {p!r} not found in overrides")
        if not isinstance(overrides[p], list):
            raise ValueError(f"Zip param {p!r} must have multiple values")

    lengths = [len(overrides[p]) for p in params]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"Zip group {params} has mismatched lengths: {dict(zip(params, lengths))}"
        )


def expand_sweep_params(
    overrides: dict[str, Any],
    zip_groups: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Expand sweep parameters into a list of job configurations.

    Takes cartesian product over sweep parameters, with optional zip groups
    that vary together instead of independently.
    """
    zip_group_lists = [
        [p.strip() for p in group.split(",")] for group in (zip_groups or [])
    ]
    zipped_params = {p for group in zip_group_lists for p in group}

    fixed = {k: v for k, v in overrides.items() if not isinstance(v, list)}

    dimensions = [
        ([k], [(v,) for v in vals])
        for k, vals in overrides.items()
        if isinstance(vals, list) and k not in zipped_params
    ]

    for params in zip_group_lists:
        _validate_zip_group(params, overrides)
        values_per_param = [overrides[p] for p in params]
        dimensions.append((params, list(zip(*values_per_param))))

    if not dimensions:
        return [fixed.copy()]

    all_keys = [keys for keys, _ in dimensions]
    all_value_tuples = [vals for _, vals in dimensions]

    jobs = []
    for combo in product(*all_value_tuples):
        job = fixed.copy()
        for keys, value_tuple in zip(all_keys, combo):
            job.update(zip(keys, value_tuple))
        jobs.append(job)

    return jobs


# =============================================================================
# Output Directories
# =============================================================================


def get_sweep_params_suffix(overrides: dict[str, Any]) -> str:
    """Generate suffix from sweep parameter names."""
    sweep_keys = [k for k, v in overrides.items() if isinstance(v, list)]
    if not sweep_keys:
        return ""
    short_names = [k.split(".")[-1] for k in sorted(sweep_keys)]
    return "_" + "_".join(short_names)


def get_output_dir(
    experiment_name: str,
    overrides: dict[str, Any],
    output_arg: str | None,
) -> Path:
    """Return the output directory path for a sweep."""
    if output_arg:
        return Path(output_arg)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = get_sweep_params_suffix(overrides)

    if suffix:
        # Sweep: outputs/{exp_name}/{sweep_params}/{timestamp}/
        return Path("outputs") / experiment_name / suffix.lstrip("_") / timestamp
    else:
        # Single run: outputs/{exp_name}/{timestamp}/
        return Path("outputs") / experiment_name / timestamp


def auto_subdir_pattern(overrides: dict[str, Any]) -> str | None:
    """
    Generate subdir pattern from ALL override parameters.

    Includes all overrides (not just sweep parameters) to ensure uniqueness
    across multiple sweep commands writing to the same output directory.
    """
    if not overrides:
        return None

    parts = []
    for key in sorted(overrides.keys()):
        short_name = key.split(".")[-1]
        parts.append(f"{short_name}{{{key}}}")

    return "_".join(parts)


def format_subdir(pattern: str, overrides: dict[str, Any]) -> str:
    """Format a subdir pattern with override values."""
    result = pattern
    for key, value in overrides.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(
                placeholder, "null" if value is None else str(value)
            )
    return result


def make_job_subdir(
    job_index: int,
    job_overrides: dict[str, Any],
    subdir_pattern: str | None,
) -> str:
    """Generate subdirectory name for a job."""
    if subdir_pattern:
        return format_subdir(subdir_pattern, job_overrides)
    return str(job_index)


def check_subdir_uniqueness(
    jobs: list[dict[str, Any]], subdir_pattern: str | None
) -> None:
    """Verify all jobs produce unique subdirectory names."""
    if subdir_pattern is None:
        return

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
