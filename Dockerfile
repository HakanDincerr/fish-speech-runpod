FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    git \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Fish Speech kur
RUN git clone https://github.com/fishaudio/fish-speech /app/fish-speech
RUN cd /app/fish-speech && pip install -e ".[stable]" -q

# Ek paketler
RUN pip install runpod fastapi uvicorn httpx huggingface_hub -q

# Referans ses dosyasını kopyala
COPY referans.mp3 /app/referans.mp3

# Handler kopyala
COPY handler.py /app/handler.py

EXPOSE 8000 8081

CMD ["python3", "-u", "/app/handler.py"]
