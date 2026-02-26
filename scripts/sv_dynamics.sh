#!/bin/bash
set -e

WORKERS=192
DEVICE=cpu
OUTPUT=outputs/sv_dynamics

# Params ordered slowest-first so long jobs get dispatched early in parallel runs.
# Dominant cost factors: max_steps (zipped with gamma), then hidden_dim.
echo "=== Offline: Full Batch (GD) + SGD (batch_size=1) ==="
python sweep.py -cn=sv_dynamics \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000,8000,5000 \
    model.hidden_dim=100,50,10 \
    model.model_seed=0,1,2,3 \
    training.batch_size=null,1 \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --device=$DEVICE \
    --output=$OUTPUT/offline

echo "=== Online: Full Batch (batch_size=500) + SGD (batch_size=1) ==="
python sweep.py -cn=sv_dynamics \
    data.online=true \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000,8000,5000 \
    model.hidden_dim=100,50,10 \
    model.model_seed=0,1,2,3 \
    training.batch_size=500,1 \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --device=$DEVICE \
    --output=$OUTPUT/online

echo "=== Done ==="
