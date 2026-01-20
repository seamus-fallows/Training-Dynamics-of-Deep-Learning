#!/bin/bash
set -e

WORKERS=40
COMMON="model.hidden_dim=10,50,100 model.gamma=0.75,1.0,1.5 max_steps=5000,10000,27000 data.noise_std=0.0,0.2 --zip=model.gamma,max_steps --workers=$WORKERS"

echo "=== Full batch training ==="
python sweep.py -cn=gph $COMMON training.batch_size=null --output=outputs/gph/fullbatch

echo "=== Mini Batch Training ==="
python sweep.py -cn=gph $COMMON training.batch_size=1,2,5,10,50 training.batch_seed=0..100 --output=outputs/gph/minibatch

echo "=== Done ==="