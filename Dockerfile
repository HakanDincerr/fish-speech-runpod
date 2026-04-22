FROM frankleeeee/sglang-omni:dev

WORKDIR /app

RUN apt-get update && apt-get install -y \
    portaudio19-dev ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# sglang_omni hangi python'da? Build sırasında yaz.
RUN python3 -c "import sglang_omni; print(sglang_omni.__file__)" > /app/sglang_omni_path.txt 2>&1 || \
    python3.12 -c "import sglang_omni; print(sglang_omni.__file__)" >> /app/sglang_omni_path.txt 2>&1 || \
    echo "NOT_FOUND" > /app/sglang_omni_path.txt

RUN python3 -c "import sys; open('/app/sglang_python.txt','w').write(sys.executable)" 2>/dev/null || \
    echo "/usr/bin/python3" > /app/sglang_python.txt

RUN cat /app/sglang_omni_path.txt && cat /app/sglang_python.txt

# Fish Speech
RUN git clone https://github.com/fishaudio/fish-speech /app/fish-speech && \
    cd /app/fish-speech && pip install -e . --no-deps -q && \
    pip install cachetools einops hydra-core loralib natsort \
    pyrootutils rich silero-vad vector-quantize-pytorch vocos \
    zstandard -q

# RunPod + API
RUN pip install runpod fastapi uvicorn httpx huggingface_hub -q

COPY referans.mp3 /app/referans.mp3
COPY handler.py /app/handler.py

EXPOSE 8000 8080

CMD ["python3", "-u", "/app/handler.py"]
