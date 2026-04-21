import runpod
import base64
import os
import sys
import time
import threading
import subprocess
import requests

sys.path.insert(0, "/app/fish-speech")

_REF_AUDIO_PATH = "/app/referans.mp3"
if os.path.exists(_REF_AUDIO_PATH):
    with open(_REF_AUDIO_PATH, "rb") as f:
        REF_AUDIO_B64 = base64.b64encode(f.read()).decode()
else:
    REF_AUDIO_B64 = os.environ.get("REF_AUDIO_B64", "")

REF_TEXT = os.environ.get(
    "REF_TEXT",
    "Merhaba iyi günler, size nasıl yardımcı olabilirim. Ben size yardım etmek için "
    "buradayım herhangi bir konunuzda size hızlı bir şekilde yardımcı olabilirim, "
    "sorularınızı bekliyorum."
)

MODEL_PATH = "/runpod-volume/checkpoints/s2-pro"
BACKEND_PORT = 8080

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
        "data": [{"id": "s2-pro", "object": "model", "created": 1234567890, "owned_by": "fish-audio"}]
    }


@app.post("/v1/audio/speech")
async def speech(request: Request):
    body = await request.json()
    text = body.get("input", "")
    ref_audio = body.get("ref_audio", REF_AUDIO_B64)
    ref_text = body.get("ref_text", REF_TEXT)

    if not text:
        return JSONResponse({"error": "input is required"}, status_code=400)

    async def generate():
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                f"http://127.0.0.1:{BACKEND_PORT}/v1/tts",
                json={
                    "text": text,
                    "references": [{"audio": ref_audio, "text": ref_text}],
                    "format": "wav",
                    "streaming": True,
                    "normalize": True,
                },
            ) as resp:
                async for chunk in resp.aiter_bytes(2048):
                    yield chunk

    return StreamingResponse(generate(), media_type="audio/wav")


def run_openai_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")


def handler(job):
    job_input = job["input"]
    text = job_input.get("text", "")
    ref_audio = job_input.get("ref_audio", REF_AUDIO_B64)
    ref_text = job_input.get("ref_text", REF_TEXT)

    if not text:
        return {"error": "text is required"}

    try:
        response = requests.post(
            f"http://127.0.0.1:{BACKEND_PORT}/v1/tts",
            json={
                "text": text,
                "references": [{"audio": ref_audio, "text": ref_text}],
                "format": "wav",
                "streaming": False,
                "normalize": True,
            },
            timeout=60
        )
        audio_b64 = base64.b64encode(response.content).decode()
        return {"audio_base64": audio_b64, "format": "wav", "sample_rate": 44100}
    except Exception as e:
        return {"error": str(e)}


def fix_torchaudio():
    path = "/app/fish-speech/fish_speech/inference_engine/reference_loader.py"
    try:
        with open(path, "r") as f:
            content = f.read()
        if "torchaudio.list_audio_backends()" in content:
            content = content.replace(
                "backends = torchaudio.list_audio_backends()",
                "backends = getattr(__import__('torchaudio'), 'list_audio_backends', lambda: [])() if True else []"
            )
            with open(path, "w") as f:
                f.write(content)
            print("✅ torchaudio fix uygulandı!")
    except Exception as e:
        print(f"⚠️ fix hatası: {e}")


def start_backend():
    """Fish Speech + SGLang backend.
    --use-sglang: token-level streaming → ~300-500ms TTFB
    --half: fp16 → daha hızlı
    """
    proc = subprocess.Popen(
        [
            "python", "/app/fish-speech/tools/api_server.py",
            "--llama-checkpoint-path", MODEL_PATH,
            "--decoder-checkpoint-path", f"{MODEL_PATH}/codec.pth",
            "--listen", f"0.0.0.0:{BACKEND_PORT}",
            "--half",
            "--use-sglang",  # SGLang backend → token-level streaming → düşük TTFB
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    for line in proc.stdout:
        print(f"[fish] {line.strip()}")


def wait_for_backend(timeout=300):
    print("⏳ SGLang backend bekleniyor...")
    for i in range(timeout // 10):
        time.sleep(10)
        try:
            r = requests.get(f"http://127.0.0.1:{BACKEND_PORT}/v1/health", timeout=5)
            if r.json().get("status") == "ok":
                print(f"✅ Backend hazır! ({(i+1)*10}sn)")
                return True
        except:
            if i % 3 == 0:
                print(f"⏳ {(i+1)*10}sn...")
    return False


def load_reference():
    if not REF_AUDIO_B64:
        print("⚠️ Referans ses yok!")
        return
    ref_bytes = base64.b64decode(REF_AUDIO_B64)
    try:
        resp = requests.post(
            f"http://127.0.0.1:{BACKEND_PORT}/v1/references/add",
            data={"id": "default", "text": REF_TEXT},
            files={"audio": ("referans.mp3", ref_bytes, "audio/mpeg")},
            timeout=30
        )
        print(f"✅ Referans ses yüklendi! ({resp.status_code})")
    except Exception as e:
        print(f"⚠️ Referans yüklenemedi: {e}")


if __name__ == "__main__":

    fix_torchaudio()

    if not os.path.exists(f"{MODEL_PATH}/codec.pth"):
        print("⏳ Model indiriliyor (~11GB)...")
        os.makedirs(MODEL_PATH, exist_ok=True)
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="fishaudio/s2-pro", local_dir=MODEL_PATH)
        print("✅ Model indirildi!")
    else:
        print("✅ Model mevcut!")

    print("🚀 Fish Speech + SGLang başlatılıyor...")
    threading.Thread(target=start_backend, daemon=True).start()

    if not wait_for_backend():
        print("❌ Backend başlamadı!")
        sys.exit(1)

    load_reference()

    threading.Thread(target=run_openai_server, daemon=True).start()
    print("✅ OpenAI API hazır: port 8000")

    print("✅ RunPod handler başlatılıyor...")
    runpod.serverless.start({"handler": handler})
