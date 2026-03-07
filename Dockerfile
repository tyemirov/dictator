FROM python:3.11.8-slim-bookworm

ARG TORCH_VERSION=2.8.0
ARG TORCHAUDIO_VERSION=2.8.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/app \
    HF_HOME=/home/app/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/home/app/.cache/huggingface/hub \
    WHISPER_CACHE_DIR=/home/app/.cache/whisper \
    TORCH_HOME=/home/app/.cache/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    espeak-ng \
    ffmpeg \
    git \
    libgomp1 \
    libsndfile1 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash app

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        torch==${TORCH_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION} && \
    python -m pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

RUN mkdir -p \
    /app/.dictator-artifacts \
    /home/app/.cache/huggingface \
    /home/app/.cache/whisper \
    /home/app/.cache/torch && \
    chown -R app:app /app /home/app

USER app

EXPOSE 50051

VOLUME ["/app/.dictator-artifacts", "/home/app/.cache/huggingface", "/home/app/.cache/whisper", "/home/app/.cache/torch"]

CMD ["python", "serve.py", "--config", "/app/config.yml", "--env-file", "/app/.env"]
