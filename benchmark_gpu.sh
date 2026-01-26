#!/bin/bash
# benchmark_gpu.sh

set -e

CONFIG="diagonal_teacher"
MAX_STEPS=10000  # Short jobs for faster benchmarking
TOTAL_JOBS=250

# Parameter sweep: 1000 batch seeds × 1 gamma = 1000 jobs
PARAMS="training.batch_seed=0..${TOTAL_JOBS} training.batch_size=5,500  max_steps=${MAX_STEPS}"

# Test different worker counts
WORKER_COUNTS=(240)

echo "=== GPU Benchmarking ==="
echo "Total jobs: ${TOTAL_JOBS}"
echo "Steps per job: ${MAX_STEPS}"
echo ""

for WORKERS in "${WORKER_COUNTS[@]}"; do
    echo "----------------------------------------"
    echo "Testing with ${WORKERS} workers (GPU)"
    echo "----------------------------------------"
    
    START=$(date +%s)
    
    python sweep.py -cn=${CONFIG} ${PARAMS} \
        --workers=${WORKERS} \
        --device=cuda \
        --no-save
    
    END=$(date +%s)
    DURATION=$((END - START))
    
    echo ""
    echo "Workers: ${WORKERS} | Duration: ${DURATION}s | Jobs/sec: $(echo "scale=2; ${TOTAL_JOBS} / ${DURATION}" | bc)"
    echo ""
    
    sleep 5
done

echo "=== GPU Benchmark Complete ==="