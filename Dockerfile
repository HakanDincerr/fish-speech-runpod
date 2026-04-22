FROM frankleeeee/sglang-omni:dev

WORKDIR /app

RUN apt-get update && apt-get install -y \
    portaudio19-dev ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# sglang_omni nerede? Build sırasında bul ve kaydet
RUN echo "=== Python paths ===" && \
    find / -name "python3*" -type f 2>/dev/null | grep -v proc | head -20 && \
    echo "=== sglang_omni location ===" && \
    find / -name "sglang_omni" -type d 2>/dev/null | grep -v proc && \
    echo "=== Which python has sglang_omni ===" && \
    for py in /usr/bin/python3 /usr/bin/python3.12 /opt/conda/bin/python3 /usr/local/bin/python3; do \
        if [ -f "$py" ]; then \
            $py -c "import sglang_omni; print('FOUND:', '$py', sglang_omni.__file__)" 2>/dev/null && break; \
        fi; \
    done

# Doğru python path'ini dosyaya yaz — handler runtime'da okuyacak
RUN for py in /usr/bin/python3 /usr/bin/python3.12 /opt/conda/bin/python3 /usr/local/bin/python3 $(which python3 2>/dev/null); do \
        if [ -f "$py" ] && $py -c "import sglang_omni" 2>/dev/null; then \
            echo $py > /app/sglang_python_path.txt && \
            echo "Saved: $py" && break; \
        fi; \
    done && \
    cat /app/sglang_python_path.txt || echo "NOT FOUND" > /app/sglang_python_path.txt

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
