FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libnuma1 libnuma-dev \
    portaudio19-dev git ffmpeg curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv -q

RUN git clone https://github.com/sgl-project-dev/sglang-omni.git /workspace/sglang-omni && \
    cd /workspace/sglang-omni && \
    uv pip install ".[s2pro]" --system

RUN pip install sgl-kernel --upgrade -q

COPY patch_hf.py /tmp/patch_hf.py
RUN python3.11 /tmp/patch_hf.py

RUN git clone https://github.com/fishaudio/fish-speech /app/fish-speech && \
    cd /app/fish-speech && \
    uv pip install -e ".[stable]" --system -q

RUN uv pip install runpod fastapi uvicorn httpx huggingface_hub --system -q

COPY referans.mp3 /app/referans.mp3
COPY handler.py /app/handler.py

EXPOSE 8000 8080

CMD ["python3.11", "-u", "/app/handler.py"]
