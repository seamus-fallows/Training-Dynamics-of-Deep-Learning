#!/bin/bash
set -e

WORKERS=192
OUTPUT=outputs/rank_sweep

echo "=== Rank Metrics Sweep ==="
python sweep.py -cn=rank_sweep \
    model.model_seed=0..10 \
    model.hidden_dim=10,50,100 \
    training.batch_size=null,1,5,50 \
    --workers=$WORKERS \
    --device=cpu \
    --output=$OUTPUT

echo "=== Rank Metrics Sweep (power-law teacher, α=1) ==="
python sweep.py -cn=rank_sweep \
    model.model_seed=0..10 \
    model.hidden_dim=10,50,100 \
    training.batch_size=null,1,5,50 \
    data.params.matrix=power_law \
    data.params.alpha=1.0 \
    data.params.scale=50.0 \
    --workers=$WORKERS \
    --device=cpu \
    --output=outputs/rank_sweep_powerlaw

echo "=== Done ==="
