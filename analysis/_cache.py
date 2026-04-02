"""Shared caching utilities for analysis scripts."""

import pickle
from pathlib import Path


def save_cache(data: dict, path: Path) -> None:
    """Pickle data to path, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_cache(path: Path) -> dict | None:
    """Load pickled cache, returning None on missing or corrupt file."""
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, KeyError, TypeError, EOFError) as e:
        print(f"Warning: Failed to load cache {path} ({e}), will recompute")
        return None


def load_fingerprinted_cache(path: Path, fingerprint: str) -> dict | None:
    """Load cache only if its stored fingerprint matches."""
    cached = load_cache(path)
    if cached is not None and cached.get("fingerprint") == fingerprint:
        return cached
    return None


def save_fingerprinted_cache(
    data: dict, path: Path, fingerprint: str,
) -> None:
    """Save cache with a fingerprint for staleness detection."""
    save_cache({"fingerprint": fingerprint, **data}, path)
