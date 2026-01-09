import time
import torch as t
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


def run_single_job(args):
    device_str, steps, job_id = args

    # Import inside worker to avoid CUDA init issues
    from dln.utils import seed_rng
    from dln.data import Dataset
    from dln.model import DeepLinearNetwork
    from dln.train import Trainer
    from dln.config import ModelConfig, DataConfig, TrainingConfig

    device = t.device(device_str)
    seed_rng(job_id)

    data_cfg = DataConfig(
        train_samples=500,
        test_samples=None,
        data_seed=0,
        online=False,
        noise_std=0.0,
        params={"matrix": "diagonal", "scale": 10.0},
    )
    dataset = Dataset(data_cfg, in_dim=5, out_dim=5)

    model_cfg = ModelConfig(
        in_dim=5,
        out_dim=5,
        num_hidden=3,
        hidden_dim=50,
        gamma=1.5,
        bias=False,
        seed=job_id,
    )
    model = DeepLinearNetwork(model_cfg)

    train_cfg = TrainingConfig(
        lr=0.0001,
        batch_size=100,
        optimizer="SGD",
        optimizer_params=None,
        criterion="MSELoss",
        batch_seed=0,
    )
    trainer = Trainer(model, train_cfg, dataset, device)

    trainer.run(max_steps=steps, evaluate_every=steps, show_progress=False)

    if device_str == "cuda":
        t.cuda.synchronize()

    return job_id


def benchmark_parallel(device_str: str, n_jobs: int, steps: int = 2000):
    args = [(device_str, steps, i) for i in range(n_jobs)]

    start = time.time()
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        list(executor.map(run_single_job, args))
    elapsed = time.time() - start

    total_steps = n_jobs * steps
    print(
        f"{device_str} x{n_jobs} workers: {elapsed:.2f}s total, {total_steps / elapsed:.0f} steps/s throughput"
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    n_jobs = 2  # Adjust to your jobs-per-GPU setting

    benchmark_parallel("cpu", n_jobs)
    benchmark_parallel("cuda", n_jobs)
