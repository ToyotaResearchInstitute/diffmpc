# Benchmarks for diffmpc

## Installation
Separate environments are used for each of the solvers to avoid dependency conflicts.
Installation instructions for each environment are below.

### diffmpc
```bash
python -m venv .diffmpc-venv && source .diffmpc-venv/bin/activate
# cd to diffmpc root directory
python -m pip install -e .

# (optional) to run diffmpc on gpu install jax with cuda support
pip install --upgrade "jax[cuda12]"
```

## mpc.pytorch
```bash
python -m venv .mpcpt-venv && source .mpcpt-venv/bin/activate
cd mpc.pytorch
python -m pip install -e .
```

## trajax
The trajax environment uses a docker container. This requires docker and the Nvidia Container Toolkit to be installed.
```bash
cd trajax_env
./build.sh
./run.sh
```

## theseus
```bash
python -m venv .theseus-venv && source .theseus-venv/bin/activate
# Theseus requires a torch (≥ 2.0.0) install, if you don't have torch installed locally, select a version compatible with your system.
python -m pip install cython torch==2.8.0
# Additionally, the official wheels of Theseus are broken. There are dependencies in scikit-sparse which can be fixed by:
sudo apt-get install -y libsuitesparse-dev
pip install "numpy==1.23.5" --force-reinstall
pip install --no-cache-dir --force-reinstall scikit-sparse

python -m pip install theseus-ai
```
# Running Benchmarks
## Reinforcement Learning
Reinforcement learning benchmarks are in `benchmarks/reinforcement-learning`.

Activate the virtual environemnt `source [solver]-venv/bin/activate` where `[solver]` corresponds to the solver evaluate. Then, run `python benchmark_<solver_name>.py`. Timing results are written to `reinforcement-learning/timing_results` and can be printed using `print_timing_results.py`.

## Imitation Learning
Reinforcement learning benchmarks are in `benchmarks/imitation-learning`.

Activate the virtual environemnt `source [solver]-venv/bin/activate` where `[solver]` corresponds to the solver evaluate.

### Execution on the CPU vs on the GPU
Pytorch solver benchmarks (`theseus` and `mpc.pytorch`) expose a command line flag for `--device`

Jax solvers (`trajax` and `diffmpc`) are forced to use the CPU by setting the `JAX_PLATFORMS` environment variable as `export JAX_PLATFORMS="cpu"`.
