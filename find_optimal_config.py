"""
Find optimal (jobs_per_gpu, metric_chunks) for your GPU.
"""

import subprocess
import sys
import time
import json
import tempfile
import shutil
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# ============ CONFIGURATION ============
CONFIG_NAME = "gph"
MAX_STEPS = 100
EVALUATE_EVERY = 10

CHUNK_OPTIONS = [1, 4, 16]
MAX_JOBS_TO_TEST = 32

SYSTEM_RESERVE_GB = 0.3

TOP_N_TO_REMEASURE = 4
NUM_RUNS_FOR_AVERAGE = 2
JOB_TIMEOUT_SECONDS = 300

# Sweep configuration
TOTAL_SWEEP_JOBS = 480
NUM_GPUS = None  # None = auto-detect
# =======================================


def get_gpu_info() -> tuple[float, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,nounits,noheader"],
        capture_output=True,
        text=True,
    )
    lines = result.stdout.strip().split("\n")
    total_gb = float(lines[0]) / 1024
    return total_gb, len(lines)


def run_single_job(args: tuple) -> tuple[bool, str, str]:
    num_chunks, output_dir = args

    cmd = [
        sys.executable,
        "run.py",
        f"-cn={CONFIG_NAME}",
        "model.hidden_dim=100",
        "model.gamma=1.5",
        f"max_steps={MAX_STEPS}",
        f"evaluate_every={EVALUATE_EVERY}",
        f"metric_chunks={num_chunks}",
        "metrics=[grad_norm_squared,trace_covariances]",
        "plotting.enabled=false",
        f"hydra.run.dir={output_dir}",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=JOB_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT", output_dir

    if result.returncode != 0:
        stderr_lower = result.stderr.lower()
        if "out of memory" in stderr_lower or "cuda error" in stderr_lower:
            return False, "OOM", output_dir
        return False, result.stderr[-300:], output_dir

    history_path = Path(output_dir) / "history.json"
    if not history_path.exists():
        return False, "No history file", output_dir

    return True, "", output_dir


def measure_cuda_overhead() -> float | None:
    """Measure CUDA context overhead by comparing torch reserved vs nvidia-smi."""
    output_dir = tempfile.mkdtemp(prefix="bench_overhead_")

    try:
        cmd = [
            sys.executable,
            "run.py",
            f"-cn={CONFIG_NAME}",
            "model.hidden_dim=100",
            "model.gamma=1.5",
            "max_steps=50",
            "evaluate_every=10",
            "metric_chunks=1",
            "metrics=[grad_norm_squared,trace_covariances]",
            "plotting.enabled=false",
            f"hydra.run.dir={output_dir}",
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        max_nvidia_smi_mb = 0
        start = time.perf_counter()
        timeout = 120

        while process.poll() is None:
            if time.perf_counter() - start > timeout:
                process.kill()
                return None

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,nounits,noheader",
                ],
                capture_output=True,
                text=True,
            )
            try:
                used_mb = float(result.stdout.strip().split("\n")[0])
                max_nvidia_smi_mb = max(max_nvidia_smi_mb, used_mb)
            except (ValueError, IndexError):
                pass
            time.sleep(0.05)

        if process.returncode != 0:
            return None

        history_path = Path(output_dir) / "history.json"
        history = json.loads(history_path.read_text())
        reserved_gb = history.get("peak_vram_reserved_gb", 0)

        nvidia_smi_gb = max_nvidia_smi_mb / 1024
        overhead = nvidia_smi_gb - reserved_gb

        return max(0.1, overhead)

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def measure_vram(num_chunks: int) -> float | None:
    """Measure peak VRAM reserved for a single job."""
    output_dir = tempfile.mkdtemp(prefix="bench_vram_")

    try:
        success, error, _ = run_single_job((num_chunks, output_dir))

        if not success:
            return None

        history_path = Path(output_dir) / "history.json"
        history = json.loads(history_path.read_text())
        return history.get("peak_vram_reserved_gb")
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def measure_parallel_jobs(num_chunks: int, num_jobs: int, num_runs: int) -> dict | None:
    """Measure wall time for num_jobs running in parallel, averaged over num_runs."""
    times = []

    for _ in range(num_runs):
        output_dir = tempfile.mkdtemp(prefix="bench_parallel_")

        try:
            start = time.perf_counter()

            with ProcessPoolExecutor(max_workers=num_jobs) as executor:
                args = [(num_chunks, f"{output_dir}/job_{i}") for i in range(num_jobs)]
                results = list(executor.map(run_single_job, args))

            wall_time = time.perf_counter() - start

            for success, error, _ in results:
                if not success:
                    if error == "OOM":
                        return {"oom": True}
                    if error == "TIMEOUT":
                        return {"timeout": True}
                    return None

            times.append(wall_time)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    return {"wall_time": sum(times) / len(times)}


def main():
    total_vram, detected_gpus = get_gpu_info()
    num_gpus = NUM_GPUS if NUM_GPUS is not None else detected_gpus
    available_vram = total_vram - SYSTEM_RESERVE_GB

    print("=" * 70)
    print("GPU CONFIGURATION OPTIMIZER")
    print("=" * 70)
    print(
        f"GPU: {total_vram:.1f}GB × {num_gpus} {'(auto-detected)' if NUM_GPUS is None else '(override)'}"
    )
    print(f"Config: {CONFIG_NAME}")
    print(f"Sweep: {TOTAL_SWEEP_JOBS} jobs")
    print("=" * 70)

    # Step 1: Measure CUDA overhead (also warms up chunks=1)
    print("\nMeasuring CUDA context overhead...", end=" ", flush=True)
    cuda_overhead = measure_cuda_overhead()
    if cuda_overhead is None:
        print("failed, using 0.4GB")
        cuda_overhead = 0.4
    else:
        print(f"{cuda_overhead:.2f}GB")

    # Step 2: Measure VRAM per chunk setting
    print("\n" + "=" * 70)
    print("PHASE 1: VRAM per chunk setting")
    print("=" * 70)
    print(f"{'chunks':<10} {'reserved':<12} {'max_jobs':<10} {'test_range':<15}")
    print("-" * 47)

    vram_by_chunks = {}
    job_range_by_chunks = {}

    for num_chunks in CHUNK_OPTIONS:
        reserved_gb = measure_vram(num_chunks)
        if reserved_gb is None:
            print(f"{num_chunks:<10} {'FAILED':<12}")
            continue

        max_jobs = min(
            MAX_JOBS_TO_TEST, int(available_vram / (reserved_gb + cuda_overhead))
        )

        if max_jobs < 1:
            print(
                f"{num_chunks:<10} {reserved_gb:<12.2f} {'-':<10} {'exceeds VRAM':<15}"
            )
            continue

        min_jobs = max(1, max_jobs // 2)

        print(
            f"{num_chunks:<10} {reserved_gb:<12.2f} {max_jobs:<10} {min_jobs}-{max_jobs}"
        )

        vram_by_chunks[num_chunks] = reserved_gb
        job_range_by_chunks[num_chunks] = (min_jobs, max_jobs)

    if not vram_by_chunks:
        print("All chunk settings failed!")
        return

    # Step 3: Build configs - sample min, mid, max jobs per chunk
    configs = []
    for num_chunks, reserved_gb in vram_by_chunks.items():
        min_jobs, max_jobs = job_range_by_chunks[num_chunks]
        mid_jobs = (min_jobs + max_jobs) // 2
        jobs_to_test = sorted(set([min_jobs, mid_jobs, max_jobs]))

        for num_jobs in jobs_to_test:
            est_vram = num_jobs * (reserved_gb + cuda_overhead)
            configs.append(
                {
                    "chunks": num_chunks,
                    "jobs": num_jobs,
                    "est_vram": est_vram,
                }
            )

    configs.sort(key=lambda x: x["est_vram"])

    if not configs:
        print(f"No configurations fit in {available_vram:.1f}GB available VRAM!")
        return

    # Step 4: Warmup parallel execution
    print("\nWarmup (parallel)...", end=" ", flush=True)
    warmup = measure_parallel_jobs(max(CHUNK_OPTIONS), 2, num_runs=1)
    if warmup and not warmup.get("oom") and not warmup.get("timeout"):
        print("done")
    else:
        print("skipped")

    # Step 5: Quick measurement (single run per config)
    print("\n" + "=" * 70)
    print("PHASE 2a: Quick measurements (1 run each)")
    print("=" * 70)
    print(
        f"{'chunks':<8} {'jobs':<6} {'est_vram':<10} {'batch_time':<12} {'throughput':<10}"
    )
    print("-" * 46)

    preliminary_results = []
    oom_threshold = None

    for cfg in configs:
        chunks, jobs, est_vram = cfg["chunks"], cfg["jobs"], cfg["est_vram"]

        # Skip if close to OOM threshold (within 5%)
        if oom_threshold is not None and est_vram >= 0.95 * oom_threshold:
            print(f"{chunks:<8} {jobs:<6} {est_vram:<10.2f} {'-':<12} {'skipped':<10}")
            continue

        measurement = measure_parallel_jobs(chunks, jobs, num_runs=1)

        if measurement is None:
            print(f"{chunks:<8} {jobs:<6} {est_vram:<10.2f} {'-':<12} {'FAILED':<10}")
            continue

        if measurement.get("oom"):
            print(f"{chunks:<8} {jobs:<6} {est_vram:<10.2f} {'-':<12} {'OOM':<10}")
            oom_threshold = est_vram
            continue

        if measurement.get("timeout"):
            print(f"{chunks:<8} {jobs:<6} {est_vram:<10.2f} {'-':<12} {'TIMEOUT':<10}")
            continue

        wall_time = measurement["wall_time"]
        throughput = jobs / wall_time
        print(
            f"{chunks:<8} {jobs:<6} {est_vram:<10.2f} {wall_time:<12.2f} {throughput:<10.2f}"
        )

        preliminary_results.append(
            {
                "chunks": chunks,
                "jobs": jobs,
                "batch_time": wall_time,
                "throughput": throughput,
            }
        )

    if not preliminary_results:
        print("\nNo valid configurations!")
        return

    # Step 6: Re-measure top candidates
    preliminary_results.sort(key=lambda x: x["throughput"], reverse=True)
    top_candidates = preliminary_results[:TOP_N_TO_REMEASURE]

    print("\n" + "=" * 70)
    print(
        f"PHASE 2b: Re-measuring top {len(top_candidates)} candidates ({NUM_RUNS_FOR_AVERAGE} runs each)"
    )
    print("=" * 70)
    print(f"{'chunks':<8} {'jobs':<6} {'batch_time':<12} {'throughput':<10}")
    print("-" * 36)

    final_results = []

    for candidate in top_candidates:
        chunks, jobs = candidate["chunks"], candidate["jobs"]

        measurement = measure_parallel_jobs(chunks, jobs, num_runs=NUM_RUNS_FOR_AVERAGE)

        if measurement is None or measurement.get("oom") or measurement.get("timeout"):
            print(f"{chunks:<8} {jobs:<6} {'-':<12} {'FAILED':<10}")
            continue

        wall_time = measurement["wall_time"]
        throughput = jobs / wall_time
        print(f"{chunks:<8} {jobs:<6} {wall_time:<12.2f} {throughput:<10.2f}")

        final_results.append(
            {
                "chunks": chunks,
                "jobs": jobs,
                "batch_time": wall_time,
            }
        )

    if not final_results:
        print("\nAll re-measurements failed!")
        return

    # Step 7: Calculate sweep times
    print("\n" + "=" * 70)
    print(f"PHASE 3: Sweep time ({TOTAL_SWEEP_JOBS} jobs, {num_gpus} GPUs)")
    print("=" * 70)
    print(f"{'chunks':<8} {'jobs/gpu':<10} {'batches':<10} {'total_time':<12}")
    print("-" * 40)

    for r in final_results:
        jobs_per_batch = num_gpus * r["jobs"]
        num_batches = math.ceil(TOTAL_SWEEP_JOBS / jobs_per_batch)
        r["total_time"] = num_batches * r["batch_time"]
        print(
            f"{r['chunks']:<8} {r['jobs']:<10} {num_batches:<10} {r['total_time']:<12.1f}"
        )

    # Step 8: Recommend best config
    best = min(final_results, key=lambda x: x["total_time"])

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print(f"  metric_chunks={best['chunks']}")
    print(f"  jobs_per_gpu={best['jobs']}")
    print(f"  estimated_time={best['total_time']:.1f}s")
    print()
    print(f"Command:")
    print(
        f"  hydra.launcher.n_jobs={num_gpus * best['jobs']} metric_chunks={best['chunks']}"
    )


if __name__ == "__main__":
    main()
