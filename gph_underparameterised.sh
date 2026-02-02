#!/bin/bash
set -e

WORKERS=160
OUTPUT=outputs/underparameterized

echo "=== Full Batch Training cpu ==="
python sweep.py -cn=gph \
    model.hidden_dim=3,4,5,6,8,10,40 \
    model.gamma=0.75,1.0,1.5 \
    max_steps=25000,50000,70000 \
    model.model_seed=10..18 \
    training.batch_size=null \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --device=cpu \
    --output=$OUTPUT

echo "=== Full Batch Training gpu ==="
python sweep.py -cn=gph \
    model.hidden_dim=160,640 \
    model.gamma=0.75,1.0,1.5 \
    max_steps=25000,50000,70000 \
    model.model_seed=10..18 \
    training.batch_size=null \
    --zip=model.gamma,max_steps \
    --workers=4 \
    --device=cuda \
    --output=$OUTPUT
echo "=== Done ==="