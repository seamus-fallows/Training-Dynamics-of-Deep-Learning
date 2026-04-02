"""Shared multiprocessing pool utilities for analysis scripts."""

import os
from multiprocessing import Pool
from typing import Callable


def run_pool(
    fn: Callable,
    tasks: list,
    *,
    n_workers: int | None = None,
    max_workers: int = 0,
    initializer: Callable | None = None,
    initargs: tuple = (),
    label: str = "Progress",
) -> None:
    """Run tasks in a multiprocessing Pool with progress printing.

    n_workers: explicit worker count (overrides max_workers).
    max_workers: upper bound on workers (0 = cpu_count).
    """
    if not tasks:
        return
    if n_workers is None:
        cpu = os.cpu_count() or 1
        limit = max_workers if max_workers > 0 else cpu
        n_workers = min(limit, len(tasks))

    with Pool(n_workers, initializer=initializer, initargs=initargs) as pool:
        for i, _ in enumerate(pool.imap_unordered(fn, tasks), 1):
            print(
                f"\r  {label}: {i}/{len(tasks)} ({100 * i / len(tasks):.0f}%)",
                end="", flush=True,
            )
    print()
