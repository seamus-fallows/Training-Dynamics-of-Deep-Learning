#!/bin/bash
set -e

NUM_GPUS=10
JOBS_PER_GPU=4

LAUNCHER="hydra/launcher=joblib hydra.launcher.n_jobs=$((NUM_GPUS * JOBS_PER_GPU)) hydra.launcher.verbose=10"
COMMON="model.hidden_dim=100,50,10 model.gamma=1.5,1.0,0.75 mode=offline data.noise_std=0.0,0.2"

echo "=== Full batch training ==="
python run.py -cn=gph -m $COMMON 'training.batch_size=null' $LAUNCHER

echo "=== Mini Batch Training ==="
python run.py -cn=gph -m $COMMON training.batch_size=1,2,5,10,50 'training.batch_seed=range(0,100)' $LAUNCHER

echo "=== Done ==="