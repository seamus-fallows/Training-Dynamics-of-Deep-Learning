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


if __name__ == "__main__":
    args = parse_args()

    # Debug: print parsed arguments
    print(f"Config: {args.config_name}")
    print(f"Comparative: {args.comparative}")
    print(f"Workers: {args.workers}")
    print(f"Device: {args.device}")
    print(f"Zip groups: {args.zip_groups}")
    print(f"Output: {args.output}")
    print(f"Subdir: {args.subdir}")
    print(f"Skip existing: {args.skip_existing}")
    print(f"Fail fast: {args.fail_fast}")
    print(f"Overrides: {parse_overrides(args.overrides)}")
