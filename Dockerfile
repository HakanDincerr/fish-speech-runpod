FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    portaudio19-dev git ffmpeg curl build-essential \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# uv kur
RUN pip install uv -q

# sglang-omni için Python 3.12 venv (README'deki gibi)
RUN python3.12 -m venv /opt/sglang-venv
RUN git clone https://github.com/sgl-project-dev/sglang-omni.git /tmp/sglang-omni && \
    cd /tmp/sglang-omni && \
    /opt/sglang-venv/bin/pip install uv -q && \
    /opt/sglang-venv/bin/uv pip install ".[s2pro]"

# Fish Speech — sistem Python (3.11)
RUN git clone https://github.com/fishaudio/fish-speech /app/fish-speech && \
    cd /app/fish-speech && \
    uv pip install -e ".[stable]" --system -q

# RunPod + API
RUN uv pip install runpod fastapi uvicorn httpx huggingface_hub --system -q

COPY referans.mp3 /app/referans.mp3
COPY handler.py /app/handler.py

EXPOSE 8000 8080

CMD ["python3", "-u", "/app/handler.py"]
