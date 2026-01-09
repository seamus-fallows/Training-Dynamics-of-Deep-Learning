#!/bin/bash
set -e

NUM_GPUS=12
JOBS_PER_GPU=8
METRIC_CHUNKS=1

SWEEP="model.hidden_dim=100,10 model.gamma=1.5,1.0,0.75 mode=offline data.noise_std=0.0,0.2"
LAUNCHER="hydra/launcher=joblib hydra.launcher.n_jobs=$((NUM_GPUS * JOBS_PER_GPU)) hydra.launcher.verbose=10"

echo "=== Full Batch ==="
python run.py -cn=gph -m $SWEEP metric_chunks=$METRIC_CHUNKS $LAUNCHER

echo "=== Mini Batch (b=1) ==="
python run.py -cn=gph -m $SWEEP training.batch_size=1 'training.batch_seed=range(50)' metric_chunks=$METRIC_CHUNKS $LAUNCHER

echo "=== Mini Batch (b=10) ==="
python run.py -cn=gph -m $SWEEP training.batch_size=10 'training.batch_seed=range(50)' metric_chunks=$METRIC_CHUNKS $LAUNCHER

echo "=== Done ==="