"""
Parquet-based I/O for sweep results.

Each sweep produces a single results.parquet file where each row is one run.
Hyperparameters are scalar columns; metric curves are List[Float64] columns.
Periodic flushing to part files provides crash resilience.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any

import polars as pl
import yaml


def _consolidate(
    results_path: Path,
    parts_dir: Path,
    param_keys: list[str] | None,
) -> None:
    """Deduplicates by param_keys (newest wins). Atomic write via tmp+rename."""
    part_files = sorted(parts_dir.glob("part_*.parquet"))
    if not part_files:
        return

    frames = []
    if results_path.exists():
        frames.append(pl.read_parquet(results_path))
    for pf in part_files:
        frames.append(pl.read_parquet(pf))

    combined = pl.concat(frames)

    if param_keys:
        # Add explicit row order so "last" is well-defined after dedup
        combined = (
            combined
            .with_row_index("_order")
            .sort("_order")
            .unique(subset=param_keys, keep="last")
            .drop("_order")
        )

    tmp_path = results_path.with_suffix(".tmp.parquet")
    combined.write_parquet(tmp_path)
    os.replace(tmp_path, results_path)
    _save_param_keys(results_path.parent, param_keys or [])
    shutil.rmtree(parts_dir)


class SweepWriter:
    """Accumulates run results and writes them as a single Parquet file.

    Workers return results to the main process, which calls add() to buffer them.
    Results are periodically flushed to temporary part files for crash resilience.
    At sweep end, all parts are consolidated into a single results.parquet.
    """

    def __init__(
        self, output_dir: Path, param_keys: list[str], flush_every: int = 100
    ):
        self.output_dir = output_dir
        self.parts_dir = output_dir / "_parts"
        self.results_path = output_dir / "results.parquet"
        self.param_keys = param_keys
        self.flush_every = flush_every
        self.buffer: list[dict[str, Any]] = []

        self._part_counter = self._count_existing_parts()

    def _count_existing_parts(self) -> int:
        if not self.parts_dir.exists():
            return 0
        existing = list(self.parts_dir.glob("part_*.parquet"))
        if not existing:
            return 0
        indices = [int(p.stem.split("_")[1]) for p in existing]
        return max(indices) + 1

    def _part_path(self, index: int) -> Path:
        return self.parts_dir / f"part_{index:06d}.parquet"

    def consolidate_parts(self) -> None:
        """Called on startup for crash recovery and at finalize."""
        _consolidate(self.results_path, self.parts_dir, self.param_keys)

    def get_completed_params(self) -> set[tuple]:
        """When there are no sweep parameters (single job), returns {()} if any results exist."""
        if not self.results_path.exists():
            return set()

        if not self.param_keys:
            # Single job case: any rows means the job is done
            df = pl.scan_parquet(self.results_path).select(pl.first()).collect()
            return {()} if len(df) > 0 else set()

        df = (
            pl.scan_parquet(self.results_path)
            .select(self.param_keys)
            .collect()
        )
        return {tuple(row) for row in df.iter_rows()}

    def add(self, job_overrides: dict[str, Any], history: dict[str, Any]) -> None:
        """Flushes to disk when buffer is full."""
        row: dict[str, Any] = {}
        for key in self.param_keys:
            row[key] = job_overrides.get(key)
        for metric_name, values in history.items():
            row[metric_name] = values
        self.buffer.append(row)

        if len(self.buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return

        df = pl.DataFrame(self.buffer)
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        part_path = self._part_path(self._part_counter)
        tmp_path = part_path.with_suffix(".tmp.parquet")
        df.write_parquet(tmp_path)
        os.replace(tmp_path, part_path)
        self._part_counter += 1
        self.buffer.clear()

    def finalize(self) -> None:
        self.flush()
        self.consolidate_parts()


class NullWriter:
    """No-op writer for --no-save mode."""

    def consolidate_parts(self) -> None:
        pass

    def get_completed_params(self) -> set[tuple]:
        return set()

    def add(self, job_overrides: dict[str, Any], history: dict[str, Any]) -> None:
        pass

    def flush(self) -> None:
        pass

    def finalize(self) -> None:
        pass


def _save_param_keys(output_dir: Path, param_keys: list[str]) -> None:
    """Sidecar JSON file — needed by load_sweep for deduplication."""
    path = output_dir / "_param_keys.json"
    with path.open("w") as f:
        json.dump(param_keys, f)


def _load_param_keys(sweep_dir: Path) -> list[str] | None:
    path = sweep_dir / "_param_keys.json"
    if not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)


def load_sweep(sweep_dir: Path) -> pl.DataFrame:
    """Consolidates any leftover part files first (handles interrupted sweeps)."""
    parts_dir = sweep_dir / "_parts"
    results_path = sweep_dir / "results.parquet"

    if parts_dir.exists():
        param_keys = _load_param_keys(sweep_dir)
        _consolidate(results_path, parts_dir, param_keys)

    return pl.read_parquet(results_path)


def load_sweep_config(sweep_dir: Path) -> dict:
    with (sweep_dir / "config.yaml").open("r") as f:
        return yaml.safe_load(f)


# =============================================================================
# Merge Sweeps
# =============================================================================


def _flatten_config(d: dict, prefix: str = "") -> dict[str, Any]:
    """Only recurses into dicts; list values (e.g. callbacks) are kept as leaves."""
    result = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_flatten_config(v, key))
        else:
            result[key] = v
    return result


def _get_nested(d: dict, dotted_key: str) -> Any:
    current = d
    for part in dotted_key.split("."):
        current = current[part]
    return current


def merge_sweeps(
    inputs: list[Path], output: Path, keep: str = "last"
) -> pl.DataFrame:
    """Handles three cases:

    - Fixed override differences: promoted to columns automatically.
    - Different sweep value ranges: concatenated directly.
    - Overlapping runs: deduplicated (keep="last" or "first").

    All inputs must have the same metric columns (list columns).
    """
    if keep not in ("last", "first"):
        raise ValueError(f"keep must be 'last' or 'first', got {keep!r}")
    if len(inputs) < 2:
        raise ValueError("Need at least 2 inputs to merge")

    # Load all inputs
    configs = []
    param_keys_list = []
    frames = []

    for path in inputs:
        if not (path / "results.parquet").exists():
            raise FileNotFoundError(f"No results.parquet in {path}")
        if not (path / "config.yaml").exists():
            raise FileNotFoundError(f"No config.yaml in {path}")

        configs.append(load_sweep_config(path))
        param_keys_list.append(_load_param_keys(path) or [])
        frames.append(load_sweep(path))

    # Step 1: Flatten configs and find keys that differ across inputs
    flat_configs = [_flatten_config(c) for c in configs]
    all_config_keys: set[str] = set()
    for fc in flat_configs:
        all_config_keys.update(fc.keys())

    promoted_keys = []
    for key in sorted(all_config_keys):
        values = [fc.get(key) for fc in flat_configs]
        first = values[0]
        if not all(v == first for v in values[1:]):
            promoted_keys.append(key)

    # Step 2: Validate metric columns (list columns must match across all inputs)
    def list_cols(df: pl.DataFrame) -> set[str]:
        return {c for c in df.columns if isinstance(df[c].dtype, pl.List)}

    def scalar_cols(df: pl.DataFrame) -> set[str]:
        return {c for c in df.columns if not isinstance(df[c].dtype, pl.List)}

    reference = list_cols(frames[0])
    for i, df in enumerate(frames[1:], 1):
        lc = list_cols(df)
        if lc != reference:
            missing = reference - lc
            extra = lc - reference
            parts = []
            if missing:
                parts.append(f"missing {missing}")
            if extra:
                parts.append(f"extra {extra}")
            raise ValueError(
                f"Metric column mismatch: {inputs[i]} has {', '.join(parts)} "
                f"compared to {inputs[0]}"
            )

    # Steps 3+4: Add missing scalar columns and promoted config-diff columns.
    # Batched into a single with_columns call per frame for efficiency.
    all_scalar = set()
    for df in frames:
        all_scalar.update(scalar_cols(df))

    for i in range(len(frames)):
        new_cols = []
        existing = set(frames[i].columns)

        # Missing scalar columns (sweep params in one input but not another)
        for col in sorted(all_scalar - existing):
            try:
                value = _get_nested(configs[i], col)
            except KeyError:
                raise ValueError(
                    f"Column {col!r} missing from {inputs[i]} and not in its config.yaml"
                )
            new_cols.append(pl.lit(value).alias(col))

        # Promoted columns (config diffs not yet in DataFrame)
        for key in promoted_keys:
            if key not in existing:
                value = flat_configs[i].get(key)
                new_cols.append(pl.lit(value).alias(key))

        if new_cols:
            frames[i] = frames[i].with_columns(new_cols)

    # Step 5: Build merged param_keys (order-preserved union)
    seen: set[str] = set()
    merged_param_keys: list[str] = []
    for pk_list in param_keys_list:
        for k in pk_list:
            if k not in seen:
                merged_param_keys.append(k)
                seen.add(k)
    for k in promoted_keys:
        if k not in seen:
            merged_param_keys.append(k)
            seen.add(k)
    for col in sorted(all_scalar):
        if col not in seen:
            merged_param_keys.append(col)
            seen.add(col)

    # Step 6: Align column order across frames, concat, and dedup
    all_columns = list(dict.fromkeys(
        col for df in frames for col in df.columns
    ))
    frames = [df.select(all_columns) for df in frames]
    combined = pl.concat(frames, how="vertical_relaxed")

    if merged_param_keys:
        combined = (
            combined
            .with_row_index("_order")
            .sort("_order")
            .unique(subset=merged_param_keys, keep=keep)
            .drop("_order")
        )

    # Step 7: Write output
    output.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(output / "results.parquet")
    _save_param_keys(output, merged_param_keys)

    # Write first input's config as reference. Promoted values are authoritative
    # from the DataFrame columns, not from this config file.
    with (output / "config.yaml").open("w") as f:
        yaml.safe_dump(configs[0], f, default_flow_style=False, sort_keys=False)

    return combined


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sweep results I/O utilities")
    subparsers = parser.add_subparsers(dest="command")

    merge_parser = subparsers.add_parser(
        "merge", help="Merge multiple sweep directories"
    )
    merge_parser.add_argument(
        "inputs", nargs="+", type=Path,
        help="Sweep directories to merge",
    )
    merge_parser.add_argument(
        "-o", "--output", required=True, type=Path,
        help="Output directory for merged results",
    )
    merge_parser.add_argument(
        "--keep", choices=["first", "last"], default="last",
        help="Dedup strategy for overlapping runs (default: last)",
    )

    args = parser.parse_args()

    if args.command == "merge":
        result = merge_sweeps(args.inputs, args.output, keep=args.keep)
        print(f"Merged {len(result)} rows into {args.output / 'results.parquet'}")
        print(f"Columns: {result.columns}")
    else:
        parser.print_help()
