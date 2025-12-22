#!/bin/bash
set -e

NUM_GPUS=12
JOBS_PER_GPU=10
METRIC_CHUNKS=2

SWEEP="model.hidden_dim=100,10 model.gamma=1.5,1.0,0.75 mode=offline,online data.noise_std=0.0,0.2"
LAUNCHER="hydra/launcher=joblib hydra.launcher.n_jobs=$((NUM_GPUS * JOBS_PER_GPU)) hydra.launcher.verbose=10"

echo "=== Full Batch ==="
python run.py -cn=gph -m $SWEEP metric_chunks=1 $LAUNCHER

echo "=== Mini Batch ==="
python run.py -cn=gph -m $SWEEP training.batch_size=5 'training.batch_seed=range(20)' metric_chunks=$METRIC_CHUNKS $LAUNCHER

echo "=== Done ==="