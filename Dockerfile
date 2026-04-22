FROM frankleeeee/sglang-omni:dev

WORKDIR /app

RUN apt-get update && apt-get install -y \
    portaudio19-dev ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# Fish Speech kur — stable extra yerine doğrudan core paketleri
RUN git clone https://github.com/fishaudio/fish-speech /app/fish-speech

# pyproject.toml'daki [stable] extra'sı bu image'da çakışıyor
# Bu yüzden --no-deps ile kur, sonra eksik bağımlılıkları elle ekle
RUN cd /app/fish-speech && pip install -e . --no-deps -q && \
    pip install \
    cachetools \
    einops \
    hydra-core \
    loralib \
    natsort \
    pyrootutils \
    rich \
    silero-vad \
    vector-quantize-pytorch \
    vocos \
    zstandard \
    -q

# RunPod + API
RUN pip install runpod fastapi uvicorn httpx huggingface_hub -q

COPY referans.mp3 /app/referans.mp3
COPY handler.py /app/handler.py

EXPOSE 8000 8080

CMD ["python3", "-u", "/app/handler.py"]
