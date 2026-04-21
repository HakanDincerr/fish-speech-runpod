FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    portaudio19-dev git ffmpeg curl build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

# Fish Speech kur
RUN git clone https://github.com/fishaudio/fish-speech /app/fish-speech
RUN cd /app/fish-speech && pip install -e ".[stable]" -q

# Torch 2.5.1 yükle (torchvision circular import fix)
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124 \
    --force-reinstall -q

# FlashInfer for torch 2.5
RUN pip install flashinfer \
    -i https://flashinfer.ai/whl/cu124/torch2.5/ -q

# SGLang
RUN pip install "sglang[all]" \
    --find-links https://flashinfer.ai/whl/cu124/torch2.5/ -q

# sglang-omni — git clone ile kur
RUN git clone https://github.com/sgl-project/sglang-omni.git /tmp/sglang-omni && \
    cd /tmp/sglang-omni && pip install -e . 2>&1 | tail -30

# Diğer bağımlılıklar
RUN pip install runpod fastapi uvicorn httpx huggingface_hub -q

COPY referans.mp3 /app/referans.mp3
COPY handler.py /app/handler.py

EXPOSE 8000 8080

CMD ["python3", "-u", "/app/handler.py"]
