#!/bin/bash
set -e

OUTPUT="hydra.sweep.dir=/tmp/extended_runs"
COMMON="mode=offline data.noise_std=0.0,0.2 model.hidden_dim=100,50,10 training.batch_size=50"
SEEDS='training.batch_seed=range(5)'
LAUNCHER="hydra/launcher=joblib hydra.launcher.n_jobs=72"

echo "=== gamma=0.75 (7000 steps) ==="
python run.py -cn=gph -m model.gamma=0.75 max_steps=7000 $COMMON "$SEEDS" $OUTPUT $LAUNCHER

echo "=== gamma=1.0 (12000 steps) ==="
python run.py -cn=gph -m model.gamma=1.0 max_steps=12000 $COMMON "$SEEDS" $OUTPUT $LAUNCHER

echo "=== gamma=1.5 (29000 steps) ==="
python run.py -cn=gph -m model.gamma=1.5 max_steps=29000 $COMMON "$SEEDS" $OUTPUT $LAUNCHER

echo "=== Done ==="
echo "Results in /tmp/extended_runs"