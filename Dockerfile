FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    portaudio19-dev git ffmpeg curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv kur
RUN pip install uv -q

# sglang-omni venv
RUN python3.11 -m venv /opt/sglang-venv

# Verbose install — tam hatayı görmek için 2>&1 olmadan
RUN git clone https://github.com/sgl-project-dev/sglang-omni.git /tmp/sglang-omni

# Önce pyproject.toml'u göster
RUN cat /tmp/sglang-omni/pyproject.toml | head -80

# s2pro extra'sını verbose kur
RUN cd /tmp/sglang-omni && \
    /opt/sglang-venv/bin/pip install --upgrade pip -q && \
    /opt/sglang-venv/bin/pip install ".[s2pro]" -v 2>&1 | tail -50 || true

# Fish Speech
RUN git clone https://github.com/fishaudio/fish-speech /app/fish-speech && \
    cd /app/fish-speech && \
    uv pip install -e ".[stable]" --system -q

# RunPod + API
RUN uv pip install runpod fastapi uvicorn httpx huggingface_hub --system -q

COPY referans.mp3 /app/referans.mp3
COPY handler.py /app/handler.py

EXPOSE 8000 8080

CMD ["python3", "-u", "/app/handler.py"]
