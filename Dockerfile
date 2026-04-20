FROM runpod/pytorch:2.5.1-py3.11-cuda12.4.1-devel-ubuntu22.04

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
RUN pip install runpod fastapi uvicorn httpx -q

# Modeli indir (~11GB, build sırasında indirilir)
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(\
    repo_id='fishaudio/s2-pro', \
    local_dir='/app/checkpoints/s2-pro'\
)"

# Handler kopyala
COPY handler.py /app/handler.py

# Port 8000 OpenAI API, 8081 Fish Speech internal
EXPOSE 8000 8081

CMD ["python3", "-u", "/app/handler.py"]
