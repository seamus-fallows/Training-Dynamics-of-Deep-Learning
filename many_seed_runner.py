#!/usr/bin/env python
"""
Run 3-dataseed experiment without Hydra sweep overhead.
"""

import os

# Set to True to force CPU training
USE_CPU = False

if USE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from itertools import islice, product
import sys
import time


OUTPUT_DIR = Path("outputs/gpu_test9")
N_WORKERS = 40

WIDTHS = [100, 50, 10]
DATA_SEEDS = [0, 1, 2, 3]
MODEL_SEEDS = [0, 1, 2, 3]
N_BATCH_SEEDS = 10

COMMON = {
    "model.gamma": 0.75,
    "data.online": False,
    "data.noise_std": 0.2,
    "data.test_samples": False,
    "metrics": [],
    "max_steps": 3000,
    "plotting.enabled": False,
}


def get_output_dir(width, batch_size, batch_seed, data_seed, model_seed):
    b_str = "None" if batch_size is None else str(batch_size)
    return (
        OUTPUT_DIR / f"h{width}_g0.75_b{b_str}_s{batch_seed}_d{data_seed}_m{model_seed}"
    )


def worker_init():
    import os
    import sys

    os.environ["TQDM_DISABLE"] = "1"

    # Suppress stderr (where tqdm writes)
    sys.stderr = open(os.devnull, "w")

    global _run
    from runner import run as _run


def run_single(args):
    width, batch_size, batch_seed, data_seed, model_seed = args

    overrides = {
        **COMMON,
        "model.hidden_dim": width,
        "model.seed": model_seed,
        "data.data_seed": data_seed,
        "training.batch_size": batch_size,
        "training.batch_seed": batch_seed,
    }
    output_dir = get_output_dir(width, batch_size, batch_seed, data_seed, model_seed)

    # Skip if already done
    if (output_dir / "history.json").exists():
        return (*args, "skipped")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        _run("gph", overrides=overrides, output_dir=output_dir, autoplot=False)
        return (*args, True)
    except Exception as e:
        return (*args, False, str(e))


def job_generator():
    # GD runs (batch_size=None, batch_seed=0)
    for width, data_seed, model_seed in product(WIDTHS, DATA_SEEDS, MODEL_SEEDS):
        yield (width, None, 0, data_seed, model_seed)

    # SGD runs
    for width, data_seed, model_seed, batch_seed in product(
        WIDTHS, DATA_SEEDS, MODEL_SEEDS, range(N_BATCH_SEEDS)
    ):
        yield (width, 5, batch_seed, data_seed, model_seed)


def print_progress(completed, total, skipped, failed, start_time):
    elapsed = time.time() - start_time
    rate = completed / elapsed if elapsed > 0 else 0
    eta = (total - completed) / rate if rate > 0 else 0
    sys.stdout.write(
        f"\rProgress: {completed}/{total} ({100 * completed / total:.1f}%) | "
        f"{rate:.1f}/s | ETA: {eta:.0f}s | skipped: {skipped} | failed: {failed}    "
    )
    sys.stdout.flush()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Count jobs
    n_gd = len(WIDTHS) * len(DATA_SEEDS) * len(MODEL_SEEDS)
    n_sgd = len(WIDTHS) * len(DATA_SEEDS) * len(MODEL_SEEDS) * N_BATCH_SEEDS
    total_jobs = n_gd + n_sgd

    print(
        f"Running {total_jobs} jobs ({n_gd} GD + {n_sgd} SGD) with {N_WORKERS} workers"
    )
    print(f"Output: {OUTPUT_DIR}")

    failed = []
    completed = 0
    skipped = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=worker_init) as pool:
        jobs = job_generator()
        futures = {}

        for job in islice(jobs, N_WORKERS):
            futures[pool.submit(run_single, job)] = job

        for job in jobs:
            done = next(as_completed(futures))
            result = done.result()
            if result[-1] == "skipped":
                skipped += 1
            elif result[-1] is not True:
                failed.append(result)
            del futures[done]
            completed += 1
            print_progress(completed, total_jobs, skipped, len(failed), start_time)
            futures[pool.submit(run_single, job)] = job

        for future in as_completed(futures):
            result = future.result()
            if result[-1] == "skipped":
                skipped += 1
            elif result[-1] is not True:
                failed.append(result)
            completed += 1
            print_progress(completed, total_jobs, skipped, len(failed), start_time)

    print(f"\n\nCompleted: {completed}/{total_jobs} in {time.time() - start_time:.1f}s")
    print(f"Skipped (already done): {skipped}")
    if failed:
        print(f"Failed: {len(failed)}")
        for *args, success, error in failed[:10]:
            print(f"  {args}: {error}")


if __name__ == "__main__":
    main()
