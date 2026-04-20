import runpod
import base64
import os
import sys
import time
import threading
import subprocess
import requests

sys.path.insert(0, "/app/fish-speech")

# Referans sesi dosyadan oku
_REF_AUDIO_PATH = "/app/referans.mp3"
if os.path.exists(_REF_AUDIO_PATH):
    with open(_REF_AUDIO_PATH, "rb") as f:
        REF_AUDIO_B64 = base64.b64encode(f.read()).decode()
else:
    REF_AUDIO_B64 = os.environ.get("REF_AUDIO_B64", "")

REF_TEXT = os.environ.get(
    "REF_TEXT",
    "Merhaba iyi günler, size nasıl yardımcı olabilirim. Ben size yardım etmek için buradayım herhangi bir konunuzda size hızlı bir şekilde yardımcı olabilirim, sorularınızı bekliyorum."
)

# FastAPI OpenAI wrapper
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import uvicorn

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [{
            "id": "s2-pro",
            "object": "model",
            "created": 1234567890,
            "owned_by": "fish-audio"
        }]
    }


@app.post("/v1/audio/speech")
async def speech(request: Request):
    body = await request.json()

    text = body.get("input", "")
    ref_audio = body.get("ref_audio", REF_AUDIO_B64)
    ref_text = body.get("ref_text", REF_TEXT)
    streaming = body.get("stream", True)

    if not text:
        return JSONResponse({"error": "input is required"}, status_code=400)

    async def generate():
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                "http://127.0.0.1:8081/v1/tts",
                json={
                    "text": text,
                    "references": [{"audio": ref_audio, "text": ref_text}],
                    "format": "wav",
                    "streaming": streaming
                },
                timeout=60
            ) as resp:
                async for chunk in resp.aiter_bytes(4096):
                    yield chunk

    if streaming:
        return StreamingResponse(generate(), media_type="audio/wav")
    else:
        audio_bytes = b""
        async for chunk in generate():
            audio_bytes += chunk
        return StreamingResponse(
            iter([audio_bytes]),
            media_type="audio/wav"
        )


def run_openai_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")


# RunPod handler
def handler(job):
    job_input = job["input"]
    text = job_input.get("text", "")
    ref_audio = job_input.get("ref_audio", REF_AUDIO_B64)
    ref_text = job_input.get("ref_text", REF_TEXT)

    if not text:
        return {"error": "text is required"}

    try:
        response = requests.post(
            "http://127.0.0.1:8081/v1/tts",
            json={
                "text": text,
                "references": [{"audio": ref_audio, "text": ref_text}],
                "format": "wav",
                "streaming": False
            },
            timeout=60
        )
        audio_b64 = base64.b64encode(response.content).decode()
        return {"audio_base64": audio_b64, "format": "wav", "sample_rate": 44100}
    except Exception as e:
        return {"error": str(e)}


def fix_torchaudio():
    """torchaudio uyumsuzluk fix'i"""
    ref_loader_path = "/app/fish-speech/fish_speech/inference_engine/reference_loader.py"
    try:
        with open(ref_loader_path, "r") as f:
            content = f.read()

        # Eski kodu yeni versiyonla değiştir
        old = """        try:
            backends = torchaudio.list_audio_backends()
            if "ffmpeg" in backends:
                self.backend = "ffmpeg"
            else:
                self.backend = "soundfile"
        except AttributeError:"""

        new = """        import torchaudio as _torchaudio
        try:
            backends = getattr(_torchaudio, "list_audio_backends", lambda: [])()
            if "ffmpeg" in backends:
                self.backend = "ffmpeg"
            else:
                self.backend = "soundfile"
        except AttributeError:"""

        if old in content:
            content = content.replace(old, new)
            with open(ref_loader_path, "w") as f:
                f.write(content)
            print("✅ torchaudio fix uygulandı!")
        else:
            print("ℹ️ torchaudio fix zaten uygulanmış veya farklı versiyon")
    except Exception as e:
        print(f"⚠️ torchaudio fix hatası: {e}")


if __name__ == "__main__":

    # 1. torchaudio fix uygula
    fix_torchaudio()

    # 2. Modeli indir (ilk çalışmada)
    MODEL_PATH = "/runpod-volume/checkpoints/s2-pro"
    if not os.path.exists(f"{MODEL_PATH}/codec.pth"):
        print("⏳ Model indiriliyor (~11GB), lütfen bekleyin...")
        os.makedirs(MODEL_PATH, exist_ok=True)
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="fishaudio/s2-pro",
            local_dir=MODEL_PATH
        )
        print("✅ Model indirildi!")
    else:
        print("✅ Model zaten mevcut, indirme atlandı!")

    # 3. Fish Speech api_server'ı arka planda başlat
    def run_fish_server():
        subprocess.run([
            "python", "/app/fish-speech/tools/api_server.py",
            "--llama-checkpoint-path", MODEL_PATH,
            "--decoder-checkpoint-path", f"{MODEL_PATH}/codec.pth",
            "--listen", "0.0.0.0:8081",
            "--half"
        ])

    fish_thread = threading.Thread(target=run_fish_server, daemon=True)
    fish_thread.start()

    print("⏳ Fish Speech sunucusu yükleniyor...")

    # Sunucu hazır olana kadar bekle
    for i in range(20):
        time.sleep(15)
        try:
            r = requests.get("http://127.0.0.1:8081/v1/health", timeout=5)
            if r.json().get("status") == "ok":
                print(f"✅ Fish Speech hazır! ({(i+1)*15}sn)")
                break
        except:
            print(f"⏳ {(i+1)*15}sn bekleniyor...")

    # 4. Referans sesi yükle
    if REF_AUDIO_B64:
        ref_bytes = base64.b64decode(REF_AUDIO_B64)
        try:
            requests.post(
                "http://127.0.0.1:8081/v1/references/add",
                data={"id": "default", "text": REF_TEXT},
                files={"audio": ("referans.mp3", ref_bytes, "audio/mpeg")}
            )
            print("✅ Referans ses yüklendi!")
        except Exception as e:
            print(f"⚠️ Referans yüklenemedi: {e}")
    else:
        print("⚠️ Referans ses bulunamadı!")

    # 5. OpenAI wrapper'ı arka planda başlat
    openai_thread = threading.Thread(target=run_openai_server, daemon=True)
    openai_thread.start()
    print("✅ OpenAI API hazır: port 8000")

    # 6. RunPod serverless handler başlat
    print("✅ RunPod handler başlatılıyor...")
    runpod.serverless.start({"handler": handler})
