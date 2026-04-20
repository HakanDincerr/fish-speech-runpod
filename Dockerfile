FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

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

# torchaudio fix
RUN python3 -c "
content = open('/app/fish-speech/fish_speech/inference_engine/reference_loader.py').read()
old = '''        try:
            backends = torchaudio.list_audio_backends()
            if \"ffmpeg\" in backends:
                self.backend = \"ffmpeg\"
            else:
                self.backend = \"soundfile\"
        except AttributeError:'''
new = '''        import torchaudio as _torchaudio
        try:
            backends = getattr(_torchaudio, \"list_audio_backends\", lambda: [])()
            if \"ffmpeg\" in backends:
                self.backend = \"ffmpeg\"
            else:
                self.backend = \"soundfile\"
        except AttributeError:'''
open('/app/fish-speech/fish_speech/inference_engine/reference_loader.py', 'w').write(content.replace(old, new))
print('✅ torchaudio fix applied')
"

# Ek paketler
RUN pip install runpod fastapi uvicorn httpx huggingface_hub -q

# Referans ses dosyasını kopyala
COPY referans.mp3 /app/referans.mp3

# Handler kopyala
COPY handler.py /app/handler.py

EXPOSE 8000 8081

CMD ["python3", "-u", "/app/handler.py"]
