# CUDA 11.8 + cuDNN 8 + compiler tools (ptxas/nvlink) included
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip python3-dev build-essential git \
    libgl1 libglew2.2 libosmesa6 libglfw3 libxrandr2 libxi6 libxinerama1 libxcursor1 \
    libsm6 libxext6 libxrender1 libglib2.0-0 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Just to be explicit; devel image already sets these
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# venv
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN pip install --upgrade pip setuptools wheel

# JAX 0.4.6 + CUDA 11.8/cuDNN 8.6
RUN pip install "jax==0.4.6" \
    "jaxlib==0.4.6+cuda11.cudnn86" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt

WORKDIR /workspace
