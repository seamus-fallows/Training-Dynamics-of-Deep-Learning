#!/bin/bash
set -e

WORKERS=32
OUTPUT=outputs/gph_offline_metrics

echo "=== Full Batch Training ==="
python sweep.py -cn=gph \
    metrics=[trace_covariances] \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000,8000,5000 \
    model.model_seed=0,1,2,3 \
    data.noise_std=0.0,0.2 \
    model.hidden_dim=100,50,10 \
    training.batch_size=null \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --device=cuda \
    --output=$OUTPUT/full_batch

echo "=== Mini Batch Training ==="
python sweep.py -cn=gph \
    metrics=[trace_covariances] \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000,8000,5000 \
    training.batch_size=50,10,5,2,1 \
    model.model_seed=0,1,2,3 \
    data.noise_std=0.0,0.2 \
    training.batch_seed=0..200 \
    model.hidden_dim=100,50,10 \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --device=cuda \
    --output=$OUTPUT/mini_batch

echo "=== Done ==="
