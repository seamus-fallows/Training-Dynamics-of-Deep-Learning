#!/bin/bash
set -e

WORKERS=96

echo "=== Test Sweep ==="
python sweep.py -cn=gph \
    model.gamma=0.75,1.0 \
    max_steps=2000,4000 \
    training.batch_size=10 \
    data.noise_std=0.0,0.2 \
    training.batch_seed=0..10000 \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --output=outputs/test

echo "=== Done ==="