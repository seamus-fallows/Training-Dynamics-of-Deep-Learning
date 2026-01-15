#!/bin/bash
set -e

NUM_GPUS=10
JOBS_PER_GPU=4

LAUNCHER="hydra/launcher=joblib hydra.launcher.n_jobs=$((NUM_GPUS * JOBS_PER_GPU)) hydra.launcher.verbose=10"
COMMON="model.hidden_dim=100,50,10 model.gamma=1.0 mode=offline data.noise_std=0.0,0.2"

echo "=== Mini Batch Training ==="
time python run.py -cn=gph -m $COMMON training.batch_size=1,10,500 "training.batch_seed=range(0,10)" $LAUNCHER

echo "=== Done ==="