#!/bin/bash
set -e

WORKERS=192
DEVICE=cpu
OUTPUT=outputs/sv_dynamics_topk

# Params ordered slowest-first so long jobs get dispatched early in parallel runs.
echo "=== Offline: Full Batch (GD) + SGD (batch_size=1) ==="
python sweep.py -cn=sv_dynamics_topk \
    data.params.k=2,3,4 \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000,8000,7000 \
    model.hidden_dim=100,50,10 \
    model.model_seed=0,1,2,3 \
    training.batch_size=null,1 \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --device=$DEVICE \
    --output=$OUTPUT/offline

echo "=== Online: Full Batch (batch_size=500) + SGD (batch_size=1) ==="
python sweep.py -cn=sv_dynamics_topk \
    data.online=true \
    data.params.k=2,3,4 \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000,8000,7000 \
    model.hidden_dim=100,50,10 \
    model.model_seed=0,1,2,3 \
    training.batch_size=500,1 \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --device=$DEVICE \
    --output=$OUTPUT/online

echo "=== Done ==="
