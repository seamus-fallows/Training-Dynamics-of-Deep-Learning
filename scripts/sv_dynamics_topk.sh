#!/bin/bash
set -e

WORKERS=192
DEVICE=cpu
OUTPUT=outputs/sv_dynamics_topk

# Group A: Fixed dim=5, varying number of active singular values (k=3,4)

echo "=== Group A Offline: Full Batch (dim=5, k=3,4) ==="
python -m dln.sweep -cn=sv_dynamics_topk_metrics \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000,10000,10000 \
    model.hidden_dim=100,50,10 \
    model.model_seed=0,1,2 \
    data.noise_std=0.0,0.2 \
    data.params.k=3,4 \
    training.batch_size=null \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --device=$DEVICE \
    --output=$OUTPUT/dim5_topk/offline/full_batch

echo "=== Group A Offline: Mini Batch (dim=5, k=3,4) ==="
python -m dln.sweep -cn=sv_dynamics_topk_metrics \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000,10000,10000 \
    model.hidden_dim=100,50,10 \
    model.model_seed=0,1,2 \
    data.noise_std=0.0,0.2 \
    data.params.k=3,4 \
    training.batch_size=1 \
    training.batch_seed=0..200 \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --device=$DEVICE \
    --output=$OUTPUT/dim5_topk/offline/mini_batch

echo "=== Group A Online: Large Batch (dim=5, k=3,4) ==="
python -m dln.sweep -cn=sv_dynamics_topk_metrics \
    data.online=true \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000,10000,10000 \
    model.hidden_dim=100,50,10 \
    model.model_seed=0,1,2 \
    data.noise_std=0.0,0.2 \
    data.params.k=3,4 \
    training.batch_size=500 \
    training.batch_seed=0..200 \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --device=$DEVICE \
    --output=$OUTPUT/dim5_topk/online/large_batch

echo "=== Group A Online: Mini Batch (dim=5, k=3,4) ==="
python -m dln.sweep -cn=sv_dynamics_topk_metrics \
    data.online=true \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000,10000,10000 \
    model.hidden_dim=100,50,10 \
    model.model_seed=0,1,2 \
    data.noise_std=0.0,0.2 \
    data.params.k=3,4 \
    training.batch_size=1 \
    training.batch_seed=0..200 \
    --zip=model.gamma,max_steps \
    --workers=$WORKERS \
    --device=$DEVICE \
    --output=$OUTPUT/dim5_topk/online/mini_batch

# Group B: Fixed k=5 active SVs, varying input/output dim (6,8,10)

echo "=== Group B Offline: Full Batch (k=5, dim=6,8,10) ==="
python -m dln.sweep -cn=sv_dynamics_topk_metrics \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000,10000,10000 \
    model.in_dim=6,8,10 \
    model.out_dim=6,8,10 \
    model.hidden_dim=100,50,10 \
    model.model_seed=0,1,2 \
    data.noise_std=0.0,0.2 \
    data.params.k=5 \
    training.batch_size=null \
    --zip=model.gamma,max_steps \
    --zip=model.in_dim,model.out_dim \
    --workers=$WORKERS \
    --device=$DEVICE \
    --output=$OUTPUT/dim_sweep_k5/offline/full_batch

echo "=== Group B Offline: Mini Batch (k=5, dim=6,8,10) ==="
python -m dln.sweep -cn=sv_dynamics_topk_metrics \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000,10000,10000 \
    model.in_dim=6,8,10 \
    model.out_dim=6,8,10 \
    model.hidden_dim=100,50,10 \
    model.model_seed=0,1,2 \
    data.noise_std=0.0,0.2 \
    data.params.k=5 \
    training.batch_size=1 \
    training.batch_seed=0..200 \
    --zip=model.gamma,max_steps \
    --zip=model.in_dim,model.out_dim \
    --workers=$WORKERS \
    --device=$DEVICE \
    --output=$OUTPUT/dim_sweep_k5/offline/mini_batch

echo "=== Group B Online: Large Batch (k=5, dim=6,8,10) ==="
python -m dln.sweep -cn=sv_dynamics_topk_metrics \
    data.online=true \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000,10000,10000 \
    model.in_dim=6,8,10 \
    model.out_dim=6,8,10 \
    model.hidden_dim=100,50,10 \
    model.model_seed=0,1,2 \
    data.noise_std=0.0,0.2 \
    data.params.k=5 \
    training.batch_size=500 \
    training.batch_seed=0..200 \
    --zip=model.gamma,max_steps \
    --zip=model.in_dim,model.out_dim \
    --workers=$WORKERS \
    --device=$DEVICE \
    --output=$OUTPUT/dim_sweep_k5/online/large_batch

echo "=== Group B Online: Mini Batch (k=5, dim=6,8,10) ==="
python -m dln.sweep -cn=sv_dynamics_topk_metrics \
    data.online=true \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000,10000,10000 \
    model.in_dim=6,8,10 \
    model.out_dim=6,8,10 \
    model.hidden_dim=100,50,10 \
    model.model_seed=0,1,2 \
    data.noise_std=0.0,0.2 \
    data.params.k=5 \
    training.batch_size=1 \
    training.batch_seed=0..200 \
    --zip=model.gamma,max_steps \
    --zip=model.in_dim,model.out_dim \
    --workers=$WORKERS \
    --device=$DEVICE \
    --output=$OUTPUT/dim_sweep_k5/online/mini_batch

echo "=== Done ==="
