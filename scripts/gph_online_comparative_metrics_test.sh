#!/bin/bash
set -e

# Test version: identical to gph_online_comparative_metrics.sh but with only
# 2 batch seeds instead of 100.  Runs each phase on CPU then GPU so you can
# compare wall-clock times.

CPU_WORKERS=192
GPU_WORKERS=64

# =============================================================================
# Large-batch model metrics
# =============================================================================

echo "=== Large-batch model metrics (CPU) ==="
time python sweep.py -cn=gph_gd_model_metrics \
    data.online=true \
    training.batch_size=500 \
    model.model_seed=0,1,2,3 \
    data.noise_std=0.0,0.2 \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000,8000,5000 \
    model.hidden_dim=100,50,10 \
    --zip=model.gamma,max_steps \
    --workers=$CPU_WORKERS \
    --device=cpu \
    --no-save 

echo ""
echo "=== Large-batch model metrics (GPU) ==="
time python sweep.py -cn=gph_gd_model_metrics \
    data.online=true \
    training.batch_size=500 \
    model.model_seed=0,1,2,3 \
    data.noise_std=0.0,0.2 \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000,8000,5000 \
    model.hidden_dim=100,50,10 \
    --zip=model.gamma,max_steps \
    --workers=$GPU_WORKERS \
    --device=cuda \
    --no-save 

# =============================================================================
# Comparative sweep: large-batch (model_a) vs mini-batch (model_b)
# =============================================================================

echo ""
echo "=== Comparative (CPU) ==="
time python sweep.py --comparative -cn=gph_metrics \
    data.online=true \
    training_a.batch_size=500 \
    model.model_seed=0,1,2,3 \
    data.noise_std=0.0,0.2 \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000,8000,5000 \
    model.hidden_dim=100,50,10 \
    training_b.batch_size=50,10,5,2,1 \
    training_b.batch_seed=0,1 \
    --zip=model.gamma,max_steps \
    --workers=$CPU_WORKERS \
    --device=cpu \
    --no-save 

echo ""
echo "=== Comparative (GPU) ==="
time python sweep.py --comparative -cn=gph_metrics \
    data.online=true \
    training_a.batch_size=500 \
    model.model_seed=0,1,2,3 \
    data.noise_std=0.0,0.2 \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000,8000,5000 \
    model.hidden_dim=100,50,10 \
    training_b.batch_size=50,10,5,2,1 \
    training_b.batch_seed=0,1 \
    --zip=model.gamma,max_steps \
    --workers=$GPU_WORKERS \
    --device=cuda \
    --no-save 

echo ""
echo "=== Done ==="
