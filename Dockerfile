FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    portaudio19-dev git ffmpeg curl build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

# Fish Speech kur
RUN git clone https://github.com/fishaudio/fish-speech /app/fish-speech
RUN cd /app/fish-speech && pip install -e ".[stable]" -q

# FlashInfer — SGLang için zorunlu (torch 2.4, CUDA 12.4)
RUN pip install flashinfer \
    -i https://flashinfer.ai/whl/cu124/torch2.4/ -q

# SGLang — Fish Speech'in --use-sglang flag'i için
RUN pip install "sglang[all]" -q

# Diğer bağımlılıklar
RUN pip install runpod fastapi uvicorn httpx huggingface_hub aiohttp -q

# Referans ses
COPY referans.mp3 /app/referans.mp3

# Handler
COPY handler.py /app/handler.py

EXPOSE 8000 8080

CMD ["python3", "-u", "/app/handler.py"]
