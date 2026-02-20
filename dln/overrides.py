"""
Override parsing and sweep expansion utilities.
"""

import re
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any


class ListValue(list):
    """A list that represents a single config value, not a sweep dimension."""


# =============================================================================
# Value Parsing
# =============================================================================


def _parse_single_value(value_str: str) -> Any:
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
        - List literals: [a,b,c] -> ListValue (single config value, not a sweep)
        - range(start, stop) or range(start, stop, step)
        - Range shorthand: start..stop or start..stop..step
        - Comma-separated values: 1,2,3 or 0.5,1.0,1.5 (sweep dimension)
        - null/None -> None
        - true/false -> bool
        - Numbers (int or float)
        - Strings
    """
    value_str = value_str.strip()

    if value_str.startswith("[") and value_str.endswith("]"):
        inner = value_str[1:-1].strip()
        if not inner:
            return ListValue()
        return ListValue(_parse_single_value(v.strip()) for v in inner.split(","))

    range_match = re.match(r"range\((\d+),\s*(\d+)(?:,\s*(\d+))?\)$", value_str)
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
    overrides = {}
    for arg in override_args:
        if "=" not in arg:
            raise ValueError(f"Invalid override format: {arg!r} (expected param=value)")
        key, value_str = arg.split("=", 1)
        overrides[key] = parse_value(value_str)
    return overrides


def split_overrides(
    overrides: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split overrides into fixed (scalar) and sweep (list-valued).

    ListValue instances (from [x,y] syntax) are treated as fixed values,
    not sweep dimensions.
    """
    fixed = {
        k: list(v) if isinstance(v, ListValue) else v
        for k, v in overrides.items()
        if not isinstance(v, list) or isinstance(v, ListValue)
    }
    sweep = {
        k: v
        for k, v in overrides.items()
        if isinstance(v, list) and not isinstance(v, ListValue)
    }
    return fixed, sweep


# =============================================================================
# Job Expansion
# =============================================================================


def _validate_zip_group(params: list[str], overrides: dict) -> None:
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
    overrides: dict[str, list[Any]],
    zip_groups: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Expand sweep parameters into a list of job configurations.

    Takes cartesian product over sweep parameters, with optional zip groups
    that vary together instead of independently.
    """
    if not overrides:
        return [{}]

    zip_group_lists = [
        [p.strip() for p in group.split(",")] for group in (zip_groups or [])
    ]
    zipped_params = {p for group in zip_group_lists for p in group}

    for params in zip_group_lists:
        _validate_zip_group(params, overrides)

    # Map each zipped param to its group index for insertion at natural position
    param_to_group = {p: i for i, group in enumerate(zip_group_lists) for p in group}
    inserted_groups: set[int] = set()

    dimensions = []
    for k, vals in overrides.items():
        if k not in zipped_params:
            dimensions.append(([k], [(v,) for v in vals]))
        elif (group_idx := param_to_group[k]) not in inserted_groups:
            inserted_groups.add(group_idx)
            params = zip_group_lists[group_idx]
            values_per_param = [overrides[p] for p in params]
            dimensions.append((params, list(zip(*values_per_param))))

    all_keys = [keys for keys, _ in dimensions]
    all_value_tuples = [vals for _, vals in dimensions]

    jobs = []
    for combo in product(*all_value_tuples):
        job = {}
        for keys, value_tuple in zip(all_keys, combo):
            job.update(zip(keys, value_tuple))
        jobs.append(job)

    return jobs


# =============================================================================
# Output Directories
# =============================================================================


def get_output_dir(experiment_name: str, output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("outputs") / experiment_name / timestamp
