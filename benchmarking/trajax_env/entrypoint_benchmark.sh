#!/bin/bash
cd /workspace || exit 1
pip install -e .
cd benchmarking/reinforcement-learning

BATCH_SIZES=(16 64 256)

for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    python benchmark_trajax.py --batch_size $BATCH_SIZE --horizon 30
done

export JAX_PLATFORMS="cpu"

for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    python benchmark_trajax.py --batch_size $BATCH_SIZE --horizon 30
done
