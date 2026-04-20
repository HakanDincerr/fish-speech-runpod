FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    portaudio19-dev git ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/fishaudio/fish-speech /app/fish-speech

# Önce fish-speech kur (kendi torch versiyonuyla)
RUN cd /app/fish-speech && pip install -e ".[stable]" -q

# Sonra torch'u uyumlu versiyonla override et
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124 \
    --force-reinstall -q

RUN pip install runpod fastapi uvicorn httpx huggingface_hub -q

COPY referans.mp3 /app/referans.mp3
COPY handler.py /app/handler.py

EXPOSE 8000 8081

CMD ["python3", "-u", "/app/handler.py"]
