#!/bin/bash
set -e

NUM_GPUS=16
JOBS_PER_GPU=12

# Order matters: leftmost varies slowest, so put slow jobs first
SWEEP="model.hidden_dim=100,10 model.gamma=1.5,1.0,0.75 mode=offline,online data.noise_std=0.0,0.2"
LAUNCHER="hydra/launcher=joblib hydra.launcher.n_jobs=$((NUM_GPUS * JOBS_PER_GPU)) hydra.launcher.verbose=10"

echo "=== Full Batch ==="
python run.py -cn=gph -m $SWEEP $LAUNCHER

echo "=== Mini Batch ==="
python run.py -cn=gph -m $SWEEP training.batch_size=5 'training.batch_seed=range(20)' $LAUNCHER

echo "=== Done ==="