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

    # Align column order across frames (parts may have been written with different ordering)
    all_columns = list(dict.fromkeys(col for df in frames for col in df.columns))
    frames = [df.select(all_columns) for df in frames]
    combined = pl.concat(frames)

    if param_keys:
        # Add explicit row order so "last" is well-defined after dedup
        combined = (
            combined.with_row_index("_order")
            .sort("_order")
            .unique(subset=param_keys, keep="last")
            .drop("_order")
        )

    # Sort by param keys for efficient predicate pushdown in downstream analysis
    sort_cols = [c for c in (param_keys or []) if c in combined.columns]
    if sort_cols:
        combined = combined.sort(sort_cols)

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

    def __init__(self, output_dir: Path, param_keys: list[str], flush_every: int = 100):
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

        df = pl.scan_parquet(self.results_path).select(self.param_keys).collect()
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


def _schema_list_cols(schema: pl.Schema) -> set[str]:
    return {name for name, dtype in schema.items() if isinstance(dtype, pl.List)}


def _schema_scalar_cols(schema: pl.Schema) -> set[str]:
    return {name for name, dtype in schema.items() if not isinstance(dtype, pl.List)}


def _has_overlapping_params(
    inputs: list[Path],
    schemas: list[pl.Schema],
    merged_param_keys: list[str],
    promoted_keys: list[str],
    flat_configs: list[dict[str, Any]],
) -> bool:
    """Check for duplicate param tuples across inputs by reading only scalar columns."""
    if not merged_param_keys:
        return False

    # Promoted columns make rows from different inputs inherently unique
    if promoted_keys:
        return False

    seen: set[tuple] = set()
    for i in range(len(inputs)):
        available = [k for k in merged_param_keys if k in schemas[i].names()]
        if not available:
            continue
        df = pl.scan_parquet(inputs[i] / "results.parquet").select(available).collect()
        for row in df.iter_rows():
            if row in seen:
                return True
            seen.add(row)
    return False


def merge_sweeps(inputs: list[Path], output: Path, keep: str = "last") -> pl.LazyFrame:
    """Handles three cases:

    - Fixed override differences: promoted to columns automatically.
    - Different sweep value ranges: concatenated directly.
    - Overlapping runs: deduplicated (keep="last" or "first").

    All inputs must have the same metric columns (list columns).

    Uses streaming I/O when possible to handle arbitrarily large merges.
    Returns a LazyFrame pointing to the written output file.
    """
    if keep not in ("last", "first"):
        raise ValueError(f"keep must be 'last' or 'first', got {keep!r}")
    if len(inputs) < 2:
        raise ValueError("Need at least 2 inputs to merge")
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")

    # Load configs and schemas (no row data loaded yet)
    configs = []
    param_keys_list = []
    schemas = []

    for path in inputs:
        if not (path / "results.parquet").exists():
            raise FileNotFoundError(f"No results.parquet in {path}")
        if not (path / "config.yaml").exists():
            raise FileNotFoundError(f"No config.yaml in {path}")

        # Consolidate any leftover parts from interrupted sweeps
        parts_dir = path / "_parts"
        if parts_dir.exists():
            pk = _load_param_keys(path)
            _consolidate(path / "results.parquet", parts_dir, pk)

        configs.append(load_sweep_config(path))
        param_keys_list.append(_load_param_keys(path) or [])
        schemas.append(pl.read_parquet_schema(path / "results.parquet"))

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
    reference = _schema_list_cols(schemas[0])
    for i, schema in enumerate(schemas[1:], 1):
        lc = _schema_list_cols(schema)
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

    # Steps 3+4: Build lazy frames with missing scalar columns and promoted config-diff columns.
    all_scalar = set()
    for schema in schemas:
        all_scalar.update(_schema_scalar_cols(schema))

    frames: list[pl.LazyFrame] = []
    for i in range(len(inputs)):
        lf = pl.scan_parquet(inputs[i] / "results.parquet")
        new_cols = []
        existing = set(schemas[i].names())

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
            lf = lf.with_columns(new_cols)
        frames.append(lf)

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
        col for schema in schemas for col in schema.names()
    ))
    for key in promoted_keys:
        if key not in all_columns:
            all_columns.append(key)
    frames = [lf.select(all_columns) for lf in frames]
    combined = pl.concat(frames, how="vertical_relaxed")

    needs_dedup = _has_overlapping_params(
        inputs, schemas, merged_param_keys, promoted_keys, flat_configs,
    )

    # Step 7: Write output
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.parquet"
    sort_cols = [c for c in merged_param_keys if c in all_columns]

    if needs_dedup:
        # Two-pass dedup: resolve duplicates using only scalar columns,
        # then collect each source's surviving rows independently.
        scalar_cols = [c for c in all_columns if c not in reference]

        # Pass 1: dedup on lightweight scalar-only frames
        scalar_frames = []
        for i, lf in enumerate(frames):
            scalar_frames.append(
                lf.select(scalar_cols)
                .with_row_index("_row_idx")
                .with_columns(pl.lit(i).cast(pl.UInt32).alias("_source"))
            )
        survivors = (
            pl.concat(scalar_frames)
            .with_row_index("_order")
            .sort("_order")
            .unique(subset=merged_param_keys, keep=keep)
            .drop("_order")
            .collect()
        )

        # Pass 2: collect surviving rows per source, write to temp parts
        tmp_parts = []
        for i, lf in enumerate(frames):
            kept = survivors.filter(pl.col("_source") == i)["_row_idx"].to_list()
            if len(kept) == 0:
                continue
            chunk = (
                lf.with_row_index("_row_idx")
                .filter(pl.col("_row_idx").is_in(kept))
                .drop("_row_idx")
                .collect()
            )
            tmp = results_path.with_name(f"_dedup_{i}.tmp.parquet")
            chunk.write_parquet(tmp)
            tmp_parts.append(tmp)
            del chunk

        merged = pl.concat([pl.scan_parquet(p) for p in tmp_parts])
        if sort_cols:
            merged = merged.sort(sort_cols)
        merged.sink_parquet(results_path)
        for p in tmp_parts:
            p.unlink()
    else:
        if sort_cols:
            combined = combined.sort(sort_cols)
        combined.sink_parquet(results_path)

    _save_param_keys(output, merged_param_keys)

    # Write first input's config as reference. Promoted values are authoritative
    # from the DataFrame columns, not from this config file.
    with (output / "config.yaml").open("w") as f:
        yaml.safe_dump(configs[0], f, default_flow_style=False, sort_keys=False)

    return pl.scan_parquet(results_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sweep results I/O utilities")
    subparsers = parser.add_subparsers(dest="command")

    merge_parser = subparsers.add_parser(
        "merge", help="Merge multiple sweep directories"
    )
    merge_parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Sweep directories to merge",
    )
    merge_parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output directory for merged results",
    )
    merge_parser.add_argument(
        "--keep",
        choices=["first", "last"],
        default="last",
        help="Dedup strategy for overlapping runs (default: last)",
    )

    args = parser.parse_args()

    if args.command == "merge":
        merge_sweeps(args.inputs, args.output, keep=args.keep)
        metadata = pl.read_parquet_schema(args.output / "results.parquet")
        n_rows = pl.scan_parquet(args.output / "results.parquet").select(pl.len()).collect().item()
        print(f"Merged {n_rows} rows into {args.output / 'results.parquet'}")
        print(f"Columns: {list(metadata.names())}")
    else:
        parser.print_help()
