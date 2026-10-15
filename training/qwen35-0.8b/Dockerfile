FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates git && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir uv==0.11.2
RUN uv pip install --system --no-cache --torch-backend=cu130 \
      torch==2.14.0 torchvision==0.29.0 \
      transformers==5.17.0 accelerate==1.15.0 \
      peft==0.21.2 bitsandbytes==0.50.2 \
      pillow==12.3.0 httpx==0.28.1 numpy==2.5.3 \
      huggingface-hub==1.31.0 safetensors==0.8.0 \
      flash-linear-attention==0.5.2 \
    && uv pip check --system \
    && uv pip freeze --system > /installed-requirements.txt

# Triton's NVIDIA launcher compiles a small host extension on first use.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Match the tokenizer patch version used for the complete corpus audit.
RUN uv pip install --system --no-cache tokenizers==0.23.2 \
    && uv pip check --system \
    && uv pip freeze --system > /installed-requirements.txt

WORKDIR /workspace
CMD ["python", "--version"]
