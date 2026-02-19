#!/bin/bash
set -e

# =============================================================================
# Vectorized Training Validation
#
# Usage:
#   bash scripts/test_vectorize.sh                    # defaults: 192 workers, auto vectorize
#   bash scripts/test_vectorize.sh 64 200             # 64 workers, vectorize=200
#   bash scripts/test_vectorize.sh 192 auto           # 192 workers, vectorize=auto
#
# Then compare:
#   python scripts/plot_vectorize_validation.py
# =============================================================================

WORKERS=192          # CPU worker processes
VECTORIZE=auto       # models per vectorized group (integer or "auto")
GPU_WORKERS=1        # worker processes for GPU vectorized (1 per GPU)
MEMORY_BUDGET=0.8    # fraction of GPU memory for auto-sizing (0.0-1.0)
OUTPUT=outputs/vectorize_test

echo "=== Config: workers=$WORKERS, vectorize=$VECTORIZE, gpu_workers=$GPU_WORKERS, memory_budget=$MEMORY_BUDGET ==="
echo "=== Cleaning previous runs ==="
rm -rf $OUTPUT

# ─────────────────────────────────────────────────────────────────────────────
# CPU baseline
# ─────────────────────────────────────────────────────────────────────────────

# echo ""
# echo "================================================================"
# echo "  CPU BASELINE (--workers=$WORKERS, --device=cpu)"
# echo "================================================================"

# echo ""
# echo "--- Full Batch (CPU) ---"
# time python sweep.py -cn=gph \
#     model.model_seed=0,1,2,3 \
#     data.noise_std=0.0,0.2 \
#     model.gamma=1.5,1.0,0.75 \
#     max_steps=26000,8000,5000 \
#     model.hidden_dim=100,50,10 \
#     training.batch_size=null \
#     --zip=model.gamma,max_steps \
#     --workers=$WORKERS \
#     --device=cpu \
#     --output=$OUTPUT/cpu/full_batch

# echo ""
# echo "--- Mini Batch (CPU) ---"
# time python sweep.py -cn=gph \
#     model.model_seed=0,1,2,3 \
#     data.noise_std=0.0,0.2 \
#     model.gamma=1.5,1.0,0.75 \
#     max_steps=26000,8000,5000 \
#     training.batch_size=50,10,5,2,1 \
#     training.batch_seed=0..100 \
#     model.hidden_dim=100,50,10 \
#     --zip=model.gamma,max_steps \
#     --workers=$WORKERS \
#     --device=cpu \
#     --output=$OUTPUT/cpu/mini_batch

echo ""
echo "================================================================"
echo "  GPU VECTORIZED (--vectorize=$VECTORIZE, --workers=$GPU_WORKERS, --device=cuda)"
echo "================================================================"

# echo ""
# echo "--- Full Batch (GPU vectorized) ---"
# time python sweep.py -cn=gph \
#     model.model_seed=0,1,2,3 \
#     data.noise_std=0.0,0.2 \
#     model.gamma=1.5,1.0,0.75 \
#     max_steps=26000,8000,5000 \
#     model.hidden_dim=100,50,10 \
#     training.batch_size=null \
#     --zip=model.gamma,max_steps \
#     --vectorize=$VECTORIZE \
#     --memory-budget=$MEMORY_BUDGET \
#     --workers=$GPU_WORKERS \
#     --device=cuda \
#     --output=$OUTPUT/gpu/full_batch

echo ""
echo "--- Mini Batch (GPU vectorized) ---"
time python sweep.py -cn=gph \
    model.model_seed=0,1,2,3 \
    data.noise_std=0.0,0.2 \
    model.gamma=1.5,1.0,0.75 \
    max_steps=26000,8000,5000 \
    training.batch_size=50,10,5,2,1 \
    training.batch_seed=0..1000 \
    model.hidden_dim=100,50,10 \
    --zip=model.gamma,max_steps \
    --vectorize=$VECTORIZE \
    --memory-budget=$MEMORY_BUDGET \
    --workers=$GPU_WORKERS \
    --device=cuda \
    --output=$OUTPUT/gpu/mini_batch

echo ""
echo "================================================================"
echo "  DONE — now run: python scripts/plot_vectorize_validation.py"
echo "================================================================"
