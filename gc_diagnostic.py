"""
Diagnostic script to measure gc.collect() and cuda.empty_cache() overhead.

Run with:
    python gc_diagnostic.py --device=cpu
    python gc_diagnostic.py --device=cuda
"""

import argparse
import gc
import time

import torch as t


def simulate_job_memory():
    """Create objects similar to what a training job would create."""
    # Simulate model
    model = t.nn.Sequential(
        t.nn.Linear(5, 100),
        t.nn.Linear(100, 100),
        t.nn.Linear(100, 100),
        t.nn.Linear(100, 5),
    )

    # Simulate data
    train_inputs = t.randn(500, 5)
    train_targets = t.randn(500, 5)

    # Simulate optimizer state
    optimizer = t.optim.SGD(model.parameters(), lr=0.01)

    # Simulate a few training steps to create gradients
    for _ in range(10):
        output = model(train_inputs)
        loss = t.nn.functional.mse_loss(output, train_targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Simulate history
    history = [{"step": i, "loss": 0.1 * i} for i in range(100)]

    return model, train_inputs, train_targets, optimizer, history


def measure_gc_overhead(num_iterations: int = 100):
    """Measure gc.collect() overhead."""
    times = []

    for i in range(num_iterations):
        # Create some garbage
        _ = simulate_job_memory()

        start = time.perf_counter()
        gc.collect()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

    return times


def measure_cuda_cache_overhead(num_iterations: int = 100):
    """Measure cuda.empty_cache() overhead."""
    if not t.cuda.is_available():
        return None

    times = []
    device = t.device("cuda")

    for i in range(num_iterations):
        # Create some GPU tensors
        x = t.randn(500, 5, device=device)
        model = t.nn.Sequential(
            t.nn.Linear(5, 100),
            t.nn.Linear(100, 100),
            t.nn.Linear(100, 100),
            t.nn.Linear(100, 5),
        ).to(device)

        # Do some work
        for _ in range(10):
            y = model(x)
            loss = y.sum()
            loss.backward()

        # Delete references
        del x, model, y, loss

        start = time.perf_counter()
        t.cuda.empty_cache()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    return times


def measure_no_cleanup(num_iterations: int = 100):
    """Measure time with no explicit cleanup."""
    times = []

    for i in range(num_iterations):
        start = time.perf_counter()
        _ = simulate_job_memory()
        # No cleanup - just let references go out of scope
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    return times


def print_stats(name: str, times: list[float]):
    """Print timing statistics."""
    print(f"\n{name}:")
    print(f"  Min:    {min(times):.2f}ms")
    print(f"  Max:    {max(times):.2f}ms")
    print(f"  Avg:    {sum(times) / len(times):.2f}ms")
    print(f"  Total:  {sum(times):.1f}ms for {len(times)} calls")
    print(f"  Projected for 50k jobs: {sum(times) / len(times) * 50000 / 1000:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    print(f"Running {args.iterations} iterations on {args.device}")
    print("=" * 50)

    # Warm up
    print("\nWarming up...")
    for _ in range(10):
        _ = simulate_job_memory()
        gc.collect()

    # Measure gc.collect()
    print("\nMeasuring gc.collect() overhead...")
    gc_times = measure_gc_overhead(args.iterations)
    print_stats("gc.collect()", gc_times)

    # Measure no cleanup
    print("\nMeasuring no-cleanup baseline...")
    no_cleanup_times = measure_no_cleanup(args.iterations)
    print_stats("No explicit cleanup", no_cleanup_times)

    # Measure cuda.empty_cache() if applicable
    if args.device == "cuda":
        if t.cuda.is_available():
            print("\nMeasuring cuda.empty_cache() overhead...")
            cuda_times = measure_cuda_cache_overhead(args.iterations)
            print_stats("cuda.empty_cache()", cuda_times)
        else:
            print("\nCUDA not available, skipping cuda.empty_cache() test")

    # Summary
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  gc.collect() avg: {sum(gc_times) / len(gc_times):.2f}ms per job")
    if args.device == "cuda" and t.cuda.is_available():
        print(
            f"  cuda.empty_cache() avg: {sum(cuda_times) / len(cuda_times):.2f}ms per job"
        )


if __name__ == "__main__":
    main()
