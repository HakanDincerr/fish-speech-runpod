FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    portaudio19-dev git ffmpeg curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv kur
RUN pip install uv -q

# sglang-omni için izole venv — sistem torch/torchvision'a dokunmaz
RUN uv venv /opt/sglang-venv -p python3.11
RUN git clone https://github.com/sgl-project-dev/sglang-omni.git /tmp/sglang-omni && \
    cd /tmp/sglang-omni && \
    /opt/sglang-venv/bin/uv pip install ".[s2pro]"

# Fish Speech — sistem Python'una kur (handler için)
RUN git clone https://github.com/fishaudio/fish-speech /app/fish-speech && \
    cd /app/fish-speech && \
    uv pip install -e ".[stable]" --system -q

# RunPod + API — sistem Python
RUN uv pip install runpod fastapi uvicorn httpx huggingface_hub --system -q

COPY referans.mp3 /app/referans.mp3
COPY handler.py /app/handler.py

EXPOSE 8000 8080

CMD ["python3", "-u", "/app/handler.py"]
