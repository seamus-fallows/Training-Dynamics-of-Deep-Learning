"""
Find optimal jobs_per_gpu and metric_chunks for your hardware.
Runs actual parallel jobs and measures real performance.
"""

import subprocess
import sys
import os
import json
import tempfile
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


# ============ EDIT THESE ============
CONFIG_NAME = "gph"
NUM_CHUNKS_OPTIONS = (1, 2, 4, 8)
JOBS_TO_TEST = (1, 2, 3, 4)
MAX_STEPS = 120
EVALUATE_EVERY = 10
START_STEP = 10
END_STEP = 110
# ====================================


def get_gpu_info() -> tuple[float, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,nounits,noheader"],
        capture_output=True,
        text=True,
    )
    lines = result.stdout.strip().split("\n")
    total = float(lines[0]) / 1024
    return total, len(lines)


def run_job(job_id: int, num_chunks: int, output_dir: str) -> tuple[bool, str]:
    """Run a single job. Returns (success, error_message)."""
    job_dir = os.path.join(output_dir, f"job_{job_id}")

    cmd = [
        sys.executable,
        "run.py",
        f"-cn={CONFIG_NAME}",
        f"max_steps={MAX_STEPS}",
        f"evaluate_every={EVALUATE_EVERY}",
        f"metric_chunks={num_chunks}",
        "plotting.enabled=false",
        f"hydra.run.dir={job_dir}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return False, result.stderr[-500:] if result.stderr else "No stderr"

    return True, ""


def read_history(job_dir: str) -> dict | None:
    """Read history.json from a job directory."""
    history_path = Path(job_dir) / "history.json"
    if not history_path.exists():
        return None
    with open(history_path) as f:
        return json.load(f)


def measure_config(num_chunks: int, num_jobs: int) -> dict | None:
    """
    Run num_jobs in parallel with given num_chunks.
    Returns {batch_time, throughput} or None on failure.
    """
    output_dir = tempfile.mkdtemp(prefix="bench_")

    try:
        with ProcessPoolExecutor(max_workers=num_jobs) as executor:
            futures = [
                executor.submit(run_job, i, num_chunks, output_dir)
                for i in range(num_jobs)
            ]
            results = [f.result() for f in futures]

        # Check for failures
        for i, (success, error) in enumerate(results):
            if not success:
                is_oom = "out of memory" in error.lower()
                return {"oom": is_oom, "error": error}

        # Read histories and extract job times
        job_times = []

        for i in range(num_jobs):
            job_dir = os.path.join(output_dir, f"job_{i}")
            history = read_history(job_dir)

            if history is None:
                return {"error": f"No history.json in {job_dir}"}

            steps = history["step"]
            timestamps = history.get("timestamp")

            if timestamps is None:
                return {"error": "No timestamp field in history"}

            try:
                start_idx = steps.index(START_STEP)
                end_idx = steps.index(END_STEP)
            except ValueError:
                return {"error": f"Steps {START_STEP} or {END_STEP} not found"}

            elapsed = timestamps[end_idx] - timestamps[start_idx]
            job_times.append(elapsed)

        # Time for batch = slowest job (all must finish)
        batch_time = max(job_times)

        # Throughput = jobs completed per second
        throughput = num_jobs / batch_time

        return {
            "batch_time": batch_time,
            "throughput": throughput,
        }

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def main():
    total_vram, num_gpus = get_gpu_info()

    print("=" * 60)
    print(f"GPU: {total_vram:.1f}GB")
    print(f"GPUs: {num_gpus}")
    print(f"Config: {CONFIG_NAME}")
    print(f"Measuring step {START_STEP} → {END_STEP}")
    print("=" * 60)
    print()

    print(
        f"{'chunks':<8} {'jobs':<6} {'batch_time':<12} {'throughput':<12} {'status':<10}"
    )
    print("-" * 48)

    results = {}

    for num_chunks in NUM_CHUNKS_OPTIONS:
        oom_hit = False

        for num_jobs in JOBS_TO_TEST:
            if oom_hit:
                print(
                    f"{num_chunks:<8} {num_jobs:<6} {'-':<12} {'-':<12} {'skipped':<10}"
                )
                continue

            data = measure_config(num_chunks, num_jobs)

            if data.get("oom"):
                print(f"{num_chunks:<8} {num_jobs:<6} {'-':<12} {'-':<12} {'OOM':<10}")
                oom_hit = True
                continue

            if "error" in data:
                print(
                    f"{num_chunks:<8} {num_jobs:<6} {'-':<12} {'-':<12} {'FAILED':<10}"
                )
                print(f"  {data['error'][:60]}")
                continue

            print(
                f"{num_chunks:<8} {num_jobs:<6} {data['batch_time']:<12.2f} {data['throughput']:<12.2f} {'OK':<10}"
            )

            results[(num_chunks, num_jobs)] = {
                "batch_time": data["batch_time"],
                "throughput": data["throughput"],
            }

        print()

    if not results:
        print("All configurations failed!")
        return

    best_key = max(results.keys(), key=lambda k: results[k]["throughput"])
    best = results[best_key]

    print("=" * 60)
    print("RECOMMENDED")
    print("=" * 60)
    print(f"  metric_chunks: {best_key[0]}")
    print(f"  jobs_per_gpu:  {best_key[1]}")
    print(f"  throughput:    {best['throughput']:.2f} jobs/sec")
    print()
    print(f"For {num_gpus} GPUs:")
    print(f"  hydra.launcher.n_jobs={num_gpus * best_key[1]}")
    print(f"  metric_chunks={best_key[0]}")


if __name__ == "__main__":
    main()
