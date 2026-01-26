#!/bin/bash
# benchmark_cpu.sh

set -e

CONFIG="diagonal_teacher"
MAX_STEPS=100  # Short jobs for faster benchmarking
TOTAL_JOBS=500

# Parameter sweep: 1000 batch seeds × 1 gamma = 1000 jobs
PARAMS="training.batch_seed=0..${TOTAL_JOBS} training.batch_size=5,500 max_steps=${MAX_STEPS}"

# Test different worker counts (adjust based on your CPU core count)
# For 128 logical cores (64 physical):
WORKER_COUNTS=(100 160 300 320)

echo "=== CPU Benchmarking ==="
echo "Total jobs: ${TOTAL_JOBS}"
echo "Steps per job: ${MAX_STEPS}"
echo ""

for WORKERS in "${WORKER_COUNTS[@]}"; do
    echo "----------------------------------------"
    echo "Testing with ${WORKERS} workers (CPU)"
    echo "----------------------------------------"
    
    START=$(date +%s)
    
    python sweep.py -cn=${CONFIG} ${PARAMS} \
        --workers=${WORKERS} \
        --device=cpu \
        --no-save \
        2>&1 | tee "benchmark_cpu_${WORKERS}workers.log"
    
    END=$(date +%s)
    DURATION=$((END - START))
    
    echo ""
    echo "Workers: ${WORKERS} | Duration: ${DURATION}s | Jobs/sec: $(echo "scale=2; ${TOTAL_JOBS} / ${DURATION}" | bc)"
    echo ""
    
    # Brief cooldown between tests
    sleep 5
done

echo "=== CPU Benchmark Complete ==="
echo "Results summary:"
grep -h "Duration:" benchmark_cpu_*.log | sort