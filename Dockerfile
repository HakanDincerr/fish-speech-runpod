FROM frankleeeee/sglang-omni:dev

WORKDIR /app

# Fish Speech kur
RUN git clone https://github.com/fishaudio/fish-speech /app/fish-speech
RUN cd /app/fish-speech && pip install -e ".[stable]" -q

# RunPod + API bağımlılıkları
RUN pip install runpod fastapi uvicorn httpx huggingface_hub -q

# Referans ses
COPY referans.mp3 /app/referans.mp3

# Handler
COPY handler.py /app/handler.py

EXPOSE 8000 8080

CMD ["python3", "-u", "/app/handler.py"]
