#!/bin/bash
set -e

WORKERS=16
DEVICE=cuda
OUTPUT=outputs/hessian_spectrum

echo "=== Offline: Full Batch (GD) + SGD (batch_size=1) ==="
python -m dln.sweep -cn=hessian_spectrum \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000,8000,5000 \
    model.hidden_dim=100,50,10 \
    training.batch_size=null,1 \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --device=$DEVICE \
    --output=$OUTPUT/offline

echo "=== Online: Full Batch (batch_size=500) + SGD (batch_size=1) ==="
python -m dln.sweep -cn=hessian_spectrum \
    data.online=true \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000,8000,5000 \
    model.hidden_dim=100,50,10 \
    training.batch_size=500,1 \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --device=$DEVICE \
    --output=$OUTPUT/online

echo "=== Done ==="
