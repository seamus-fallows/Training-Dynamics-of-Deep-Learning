#!/bin/bash
set -e

WORKERS=400
OUTPUT=outputs/gph_offline

echo "=== Full Batch Training ==="
python sweep.py -cn=gph \
    model.gamma=1.5,1.0,0.75 \
    max_steps=27000,9000,6000\
    model.hidden_dim=100,50,10 \
    model.model_seed=0,1 \
    data.noise_std=0.0,0.2 \
    training.batch_size=null \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --device=cpu \
    --output=$OUTPUT/full_batch

echo "=== Mini Batch Training (gamma=0.75) ==="
python sweep.py -cn=gph \
    model.gamma=0.75 \
    max_steps=6000 \
    model.hidden_dim=100,50,10 \
    model.model_seed=0,1 \
    data.noise_std=0.0,0.2 \
    training.batch_seed=0..10000 \
    training.batch_size=50,10,5,2,1 \
    --workers=$WORKERS \
    --device=cpu \
    --output=$OUTPUT/mini_batch_g0.75

echo "=== Mini Batch Training (gamma=1.0) ==="
python sweep.py -cn=gph \
    model.gamma=1.0 \
    max_steps=9000 \
    model.hidden_dim=100,50,10 \
    model.model_seed=0,1 \
    data.noise_std=0.0,0.2 \
    training.batch_seed=0..10000 \
    training.batch_size=50,10,5,2,1 \
    --workers=$WORKERS \
    --device=cpu \
    --output=$OUTPUT/mini_batch_g1.0

echo "=== Mini Batch Training (gamma=1.5) ==="
python sweep.py -cn=gph \
    model.gamma=1.5 \
    max_steps=27000 \
    model.hidden_dim=100,50,10 \
    model.model_seed=0,1 \
    data.noise_std=0.0,0.2 \
    training.batch_seed=0..10000 \
    training.batch_size=50,10,5,2,1 \
    --workers=$WORKERS \
    --device=cpu \
    --output=$OUTPUT/mini_batch_g1.5

echo "=== Compressing results ==="
tar -czf outputs/gph_offline.tar.gz -C outputs gph_offline

echo "=== Done ==="