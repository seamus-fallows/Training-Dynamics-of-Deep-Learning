#!/bin/bash
set -e

WORKERS=250
DEVICE=cpu
OUTPUT=outputs/sv_dynamics_topk

echo "=== Offline: Full Batch (GD) ==="
python -m dln.sweep -cn=sv_dynamics_topk \
    data.params.k=2,3,4 \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000 \
    model.hidden_dim=100,50,10 \
    model.model_seed=0,1,2 \
    training.batch_size=null \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --device=$DEVICE \
    --output=$OUTPUT/offline/full_batch

echo "=== Offline: Mini Batch (batch_size=1) ==="
python -m dln.sweep -cn=sv_dynamics_topk \
    data.params.k=2,3,4 \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000 \
    model.hidden_dim=100,50,10 \
    model.model_seed=0,1,2 \
    training.batch_size=1 \
    training.batch_seed=0..5000 \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --device=$DEVICE \
    --output=$OUTPUT/offline/mini_batch

echo "=== Online: Large Batch (batch_size=500) ==="
python -m dln.sweep -cn=sv_dynamics_topk \
    data.online=true \
    data.params.k=2,3,4 \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000 \
    model.hidden_dim=100,50,10 \
    model.model_seed=0,1,2 \
    training.batch_size=500 \
    training.batch_seed=0..5000 \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --device=$DEVICE \
    --output=$OUTPUT/online/large_batch

echo "=== Online: Mini Batch (batch_size=1) ==="
python -m dln.sweep -cn=sv_dynamics_topk \
    data.online=true \
    data.params.k=2,3,4 \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000 \
    model.hidden_dim=100,50,10 \
    model.model_seed=0,1,2 \
    training.batch_size=1 \
    training.batch_seed=0..5000 \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --device=$DEVICE \
    --output=$OUTPUT/online/mini_batch

echo "=== Done ==="
