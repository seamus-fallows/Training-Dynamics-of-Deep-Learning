#!/bin/bash
set -e

# ── Shared sweep configuration ──────────────────────────────────────
CONFIG=sv_dynamics
WORKERS=192
DEVICE=cpu
OUTPUT=outputs/sv_dynamics

GAMMA=1.5,1.0,0.75
MAX_STEPS=26000,8000,5000
HIDDEN_DIM=100,50,10
MODEL_SEED=0,1,2,3
BATCH_SEED=0..200

# Common args shared by every sweep invocation.
COMMON=(
    -cn=$CONFIG
    model.gamma=$GAMMA
    max_steps=$MAX_STEPS
    model.hidden_dim=$HIDDEN_DIM
    model.model_seed=$MODEL_SEED
    --zip=model.gamma,max_steps
    --workers=$WORKERS
    --device=$DEVICE
)

# Params ordered slowest-first so long jobs get dispatched early in parallel runs.
echo "=== Offline: Full Batch (GD) ==="
python -m dln.sweep "${COMMON[@]}" \
    training.batch_size=null \
    --output=$OUTPUT/offline/full_batch

echo "=== Offline: Mini Batch (batch_size=1) ==="
python -m dln.sweep "${COMMON[@]}" \
    training.batch_size=1 \
    training.batch_seed=$BATCH_SEED \
    --output=$OUTPUT/offline/mini_batch

echo "=== Online: Large Batch (batch_size=500) ==="
python -m dln.sweep "${COMMON[@]}" \
    data.online=true \
    training.batch_size=500 \
    training.batch_seed=$BATCH_SEED \
    --output=$OUTPUT/online/large_batch

echo "=== Online: Mini Batch (batch_size=1) ==="
python -m dln.sweep "${COMMON[@]}" \
    data.online=true \
    training.batch_size=1 \
    training.batch_seed=$BATCH_SEED \
    --output=$OUTPUT/online/mini_batch

echo "=== Done ==="
