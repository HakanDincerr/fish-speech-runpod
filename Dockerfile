FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libnuma1 libnuma-dev \
    portaudio19-dev git ffmpeg curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv -q

# sglang-omni kur
RUN git clone https://github.com/sgl-project-dev/sglang-omni.git /workspace/sglang-omni && \
    cd /workspace/sglang-omni && \
    uv pip install ".[s2pro]" --system

# sgl_kernel yeniden kur (libnuma artık mevcut)
RUN pip install sgl-kernel --upgrade -q

# torchvision::nms hatası: transformers → image_utils → torchvision zinciri
# Fix: hf.py'deki no_init_weights import'unu doğrudan patch et (torchvision'ı import etmez)
RUN python3.11 -c "
path = '/usr/local/lib/python3.11/dist-packages/sglang_omni/utils/hf.py'
with open(path) as f:
    content = f.read()

# transformers import'larını kaldır, no_init_weights'i basit context manager ile değiştir
new_content = '''from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

@contextmanager
def no_init_weights(_enable=True):
    yield

def load_pretrained_model_state_dict(model, state_dict, strict=True):
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        logger.warning(f'Missing keys: {missing}')
    if unexpected:
        logger.warning(f'Unexpected keys: {unexpected}')
    return model
'''

# Sadece transformers importları ile başlayan kısmı değiştir
import re
new_content2 = re.sub(
    r'try:.*?from transformers.*?no_init_weights.*?except.*?from transformers.*?no_init_weights',
    'pass  # patched',
    content,
    flags=re.DOTALL
)

with open(path, 'w') as f:
    f.write(new_content)
print('hf.py patched successfully!')
"

# Fish Speech
RUN git clone https://github.com/fishaudio/fish-speech /app/fish-speech && \
    cd /app/fish-speech && \
    uv pip install -e ".[stable]" --system -q

RUN uv pip install runpod fastapi uvicorn httpx huggingface_hub --system -q

COPY referans.mp3 /app/referans.mp3
COPY handler.py /app/handler.py

EXPOSE 8000 8080

CMD ["python3.11", "-u", "/app/handler.py"]
