#!/bin/bash

BATCH_SIZES=(16 64 256)
DEVICES=("cpu" "cuda")
unset JAX_PLATFORMS

for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    ../.diffmpc-venv/bin/python benchmark_diffmpc.py --batch_size $BATCH_SIZE --horizon 30

    # Run mpcpytorch and theseus for each device
    for DEVICE in "${DEVICES[@]}"; do
        ../.mpcpt-venv/bin/python benchmark_mpcpytorch.py --batch_size $BATCH_SIZE --horizon 30 --device $DEVICE
        ../.theseus-venv/bin/python benchmark_theseus.py --batch_size $BATCH_SIZE --horizon 30 --device $DEVICE
    done
done

cd ../trajax_env && ./run_and_benchmark.sh

cd ../reinforcement-learning

# diffmpc cpu
export JAX_PLATFORMS="cpu"
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    ../.diffmpc-venv/bin/python benchmark_diffmpc.py --batch_size $BATCH_SIZE --horizon 30
done
