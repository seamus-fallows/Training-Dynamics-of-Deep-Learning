#!/bin/bash
set -e

WORKERS=400
OUTPUT=outputs/gph_sweep

echo "=== Full Batch Training ==="
python sweep.py -cn=gph \
    model.hidden_dim=10,50,100 \
    model.gamma=0.75,1.0,1.5 \
    max_steps=4000,6000,25000 \
    model.model_seed=0,1 \
    data.noise_std=0.0,0.2 \
    training.batch_size=null \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --device=cpu \
    --output=$OUTPUT

echo "=== Mini Batch Training (gamma=0.75) ==="
python sweep.py -cn=gph \
    model.hidden_dim=10,50,100 \
    model.gamma=0.75 \
    max_steps=4000 \
    model.model_seed=0,1 \
    data.noise_std=0.0,0.2 \
    training.batch_size=1,2,5,10,50 \
    training.batch_seed=0..10000 \
    --workers=$WORKERS \
    --device=cpu \
    --output=$OUTPUT

echo "=== Mini Batch Training (gamma=1.0) ==="
python sweep.py -cn=gph \
    model.hidden_dim=10,50,100 \
    model.gamma=1.0 \
    max_steps=6000 \
    model.model_seed=0,1 \
    data.noise_std=0.0,0.2 \
    training.batch_size=1,2,5,10,50 \
    training.batch_seed=0..10000 \
    --workers=$WORKERS \
    --device=cpu \
    --output=$OUTPUT

echo "=== Mini Batch Training (gamma=1.5) ==="
python sweep.py -cn=gph \
    model.hidden_dim=10,50,100 \
    model.gamma=1.5 \
    max_steps=25000 \
    model.model_seed=0,1 \
    data.noise_std=0.0,0.2 \
    training.batch_size=1,2,5,10,50 \
    training.batch_seed=0..1000 \
    --workers=$WORKERS \
    --device=cpu \
    --output=$OUTPUT


echo "=== Done ==="