FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Ubuntu 24.04 → Python 3.12 dahil
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3.12-dev \
    python3-pip \
    portaudio19-dev git ffmpeg curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# pip + uv
RUN python3.12 -m pip install uv --break-system-packages -q

# sglang-omni için Python 3.12 venv
RUN python3.12 -m venv /opt/sglang-venv && \
    /opt/sglang-venv/bin/pip install uv -q

RUN git clone https://github.com/sgl-project-dev/sglang-omni.git /tmp/sglang-omni && \
    cd /tmp/sglang-omni && \
    /opt/sglang-venv/bin/uv pip install ".[s2pro]"

# PyTorch — Fish Speech için
RUN /opt/sglang-venv/bin/pip install uv -q
RUN uv pip install \
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124 \
    --system -q

# Fish Speech
RUN git clone https://github.com/fishaudio/fish-speech /app/fish-speech && \
    cd /app/fish-speech && \
    uv pip install -e ".[stable]" --system -q

# RunPod + API
RUN uv pip install runpod fastapi uvicorn httpx huggingface_hub --system -q

COPY referans.mp3 /app/referans.mp3
COPY handler.py /app/handler.py

EXPOSE 8000 8080

CMD ["python3.12", "-u", "/app/handler.py"]
