FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3.12-venv \
    portaudio19-dev git ffmpeg curl build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

# pip kur
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# uv kur (hızlı paket yöneticisi)
RUN pip install uv -q

# Python 3.12 varsayılan yap
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

# sglang-omni — DOĞRU REPO: sgl-project-dev (dev subdomain!)
# Kurulum: uv pip install ".[s2pro]" — README'den birebir
RUN git clone https://github.com/sgl-project-dev/sglang-omni.git /tmp/sglang-omni && \
    cd /tmp/sglang-omni && \
    uv pip install ".[s2pro]" --system -v

# Fish Speech kur
RUN git clone https://github.com/fishaudio/fish-speech /app/fish-speech && \
    cd /app/fish-speech && \
    uv pip install -e ".[stable]" --system -q

# RunPod + API
RUN uv pip install runpod fastapi uvicorn httpx huggingface_hub --system -q

# Referans ses
COPY referans.mp3 /app/referans.mp3

# Handler
COPY handler.py /app/handler.py

EXPOSE 8000 8080

CMD ["python3", "-u", "/app/handler.py"]
