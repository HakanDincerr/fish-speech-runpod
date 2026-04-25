FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    libnuma1 libnuma-dev \
    portaudio19-dev git ffmpeg curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv kur
RUN pip install uv -q

# sglang-omni kur — bu torch 2.9.1 + torchvision 0.24.1 getirecek
RUN git clone https://github.com/sgl-project-dev/sglang-omni.git /workspace/sglang-omni && \
    cd /workspace/sglang-omni && \
    uv pip install ".[s2pro]" --system

# torchvision NMS binary uyumsuzluğunu düzelt:
# sglang-omni'nin kurduğu torch 2.9.1 ile uyumlu torchvision kur
RUN pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu124 \
    --force-reinstall -q

# sgl_kernel yeniden kur (libnuma artık mevcut, torch 2.9.1 ile uyumlu)
RUN pip install sgl-kernel --upgrade -q

# Fish Speech kur
RUN git clone https://github.com/fishaudio/fish-speech /app/fish-speech && \
    cd /app/fish-speech && \
    uv pip install -e ".[stable]" --system -q

# RunPod + API
RUN uv pip install runpod fastapi uvicorn httpx huggingface_hub --system -q

COPY referans.mp3 /app/referans.mp3
COPY handler.py /app/handler.py

EXPOSE 8000 8080

CMD ["python3.11", "-u", "/app/handler.py"]
