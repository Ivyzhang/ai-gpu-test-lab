# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python
WORKDIR /workspace

# Install python deps in their own layer
COPY requirements.txt  requirements-gpu-linux.txt ./
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt -r requirements-gpu-linux.txt

COPY . .

# Run as non-root; the NVIDIA runtime exposes GPU devices to unpriviliged users too.
RUN useradd --create-home --shell /bin/bash appuser && chown -R appuser:appuser /workspace
USER appuser

CMD ["pytest", "-q"]
