import runpod
import base64
import os
import sys
import time
import threading
import subprocess
import requests

sys.path.insert(0, "/app/fish-speech")

# Referans ses
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

    # Streaming = düşük TTFB
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


def start_sglang_backend():
    """SGLang-Omni: Fish Speech için optimize edilmiş, 30+ tok/sn"""
    config_path = "/app/s2pro_config.yaml"
    with open(config_path, "w") as f:
        f.write(f"""
model_config:
  model_type: dual_ar
  llama_checkpoint_path: {MODEL_PATH}
  decoder_checkpoint_path: {MODEL_PATH}/codec.pth
  device: cuda
  precision: half
  compile: false

runtime_config:
  mem_fraction_static: 0.65

serving_config:
  host: 0.0.0.0
  port: {BACKEND_PORT}
""")

    proc = subprocess.Popen(
        ["python", "-m", "sglang_omni.launch_server", "--config", config_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    for line in proc.stdout:
        print(f"[sglang] {line.strip()}")


def wait_for_backend(timeout=300):
    print(f"⏳ SGLang backend bekleniyor...")
    for i in range(timeout // 10):
        time.sleep(10)
        try:
            r = requests.get(f"http://127.0.0.1:{BACKEND_PORT}/v1/health", timeout=5)
            if r.json().get("status") == "ok":
                print(f"✅ SGLang hazır! ({(i+1)*10}sn)")
                return True
        except:
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

    # 1. torchaudio fix
    fix_torchaudio()

    # 2. Model indir (yoksa)
    if not os.path.exists(f"{MODEL_PATH}/codec.pth"):
        print("⏳ Model indiriliyor (~11GB)...")
        os.makedirs(MODEL_PATH, exist_ok=True)
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="fishaudio/s2-pro", local_dir=MODEL_PATH)
        print("✅ Model indirildi!")
    else:
        print("✅ Model mevcut!")

    # 3. SGLang-Omni başlat
    print("🚀 SGLang-Omni başlatılıyor...")
    backend_thread = threading.Thread(target=start_sglang_backend, daemon=True)
    backend_thread.start()

    # 4. Hazır olana kadar bekle
    if not wait_for_backend():
        print("❌ SGLang başlamadı! Logları kontrol et.")
        sys.exit(1)

    # 5. Referans ses yükle
    load_reference()

    # 6. OpenAI wrapper
    threading.Thread(target=run_openai_server, daemon=True).start()
    print("✅ OpenAI API hazır: port 8000")

    # 7. RunPod handler
    print("✅ RunPod handler başlatılıyor...")
    runpod.serverless.start({"handler": handler})
