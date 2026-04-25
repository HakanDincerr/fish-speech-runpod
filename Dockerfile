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

# hf.py'deki sadece transformers import satirini patch et, diger fonksiyonlar kalsin
RUN cat > /tmp/fix_hf.py << 'PYEOF'
import re
path = '/usr/local/lib/python3.11/dist-packages/sglang_omni/utils/hf.py'
with open(path) as f:
    content = f.read()
stub = 'from contextlib import contextmanager\n\n@contextmanager\ndef no_init_weights(_enable=True):\n    yield\n\n'
content = re.sub(r'try:\s*\n\s*from transformers.*?no_init_weights.*?\n.*?no_init_weights', stub, content, flags=re.DOTALL)
with open(path, 'w') as f:
    f.write(content)
print('hf.py patched!')
PYEOF

RUN python3.11 /tmp/fix_hf.py

RUN git clone https://github.com/fishaudio/fish-speech /app/fish-speech && \
    cd /app/fish-speech && \
    uv pip install -e ".[stable]" --system -q

RUN uv pip install runpod fastapi uvicorn httpx huggingface_hub --system -q

COPY referans.mp3 /app/referans.mp3
COPY handler.py /app/handler.py

EXPOSE 8000 8080

CMD ["python3.11", "-u", "/app/handler.py"]
