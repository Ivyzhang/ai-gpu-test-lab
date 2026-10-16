FROM python:3.11-slim-bookworm
ARG VLLM_SPEC=vllm
ARG BNB_SPEC=bitsandbytes
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN python -m pip install --upgrade pip && \
    python -m pip install "$VLLM_SPEC" "$BNB_SPEC" accelerate huggingface_hub fastapi uvicorn requests numpy && \
    python -m pip check && python -m pip freeze > /opt/resolved-requirements.txt
WORKDIR /workspace
CMD ["bash"]
