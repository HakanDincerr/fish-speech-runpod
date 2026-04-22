"""
Fish Speech S2-Pro + SGLang-Omni
SGLang-Omni izole venv'de çalışıyor: /opt/sglang-venv
Hedef: H100 SXM'de ~150-250ms TTFA, streaming ile 300-500ms kullanıcı deneyimi
"""
import runpod
import base64
import os
import sys
import time
import threading
import subprocess
import requests
import tempfile

MODEL_PATH = os.environ.get("MODEL_PATH", "/runpod-volume/checkpoints/s2-pro")
BACKEND_PORT = 8080
SGLANG_PYTHON = "/opt/sglang-venv/bin/python"

# Referans ses
REF_AUDIO_FILE = "/app/referans.mp3"
if os.path.exists(REF_AUDIO_FILE):
    with open(REF_AUDIO_FILE, "rb") as f:
        REF_AUDIO_B64 = base64.b64encode(f.read()).decode()
else:
    REF_AUDIO_B64 = os.environ.get("REF_AUDIO_B64", "")

REF_TEXT = os.environ.get(
    "REF_TEXT",
    "Merhaba iyi günler, size nasıl yardımcı olabilirim. Ben size yardım etmek için "
    "buradayım herhangi bir konunuzda size hızlı bir şekilde yardımcı olabilirim, "
    "sorularınızı bekliyorum."
)

# sglang-omni config — repo içindeki örnek config'i kullan
SGLANG_CONFIG = "/tmp/sglang-omni/examples/configs/s2pro_tts.yaml"

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
    ref_text = body.get("ref_text", REF_TEXT)

    ref_audio_b64 = body.get("ref_audio", None)
    if ref_audio_b64:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.write(base64.b64decode(ref_audio_b64))
        tmp.close()
        ref_file = tmp.name
    else:
        ref_file = REF_AUDIO_FILE

    if not text:
        return JSONResponse({"error": "input is required"}, status_code=400)

    t_start = time.time()
    first_chunk = True

    async def generate():
        nonlocal first_chunk
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                f"http://127.0.0.1:{BACKEND_PORT}/v1/audio/speech",
                json={
                    "input": text,
                    "references": [{"audio_path": ref_file, "text": ref_text}],
                },
            ) as resp:
                async for chunk in resp.aiter_bytes(2048):
                    if first_chunk:
                        ttfb = (time.time() - t_start) * 1000
                        print(f"⚡ TTFB: {ttfb:.0f}ms | '{text[:40]}'")
                        first_chunk = False
                    yield chunk

    return StreamingResponse(generate(), media_type="audio/wav")


def run_openai_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")


def handler(job):
    job_input = job["input"]
    text = job_input.get("text", "")
    ref_text = job_input.get("ref_text", REF_TEXT)

    if not text:
        return {"error": "text is required"}

    ref_audio_b64 = job_input.get("ref_audio", None)
    if ref_audio_b64:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.write(base64.b64decode(ref_audio_b64))
        tmp.close()
        ref_file = tmp.name
    else:
        ref_file = REF_AUDIO_FILE

    t_start = time.time()
    try:
        response = requests.post(
            f"http://127.0.0.1:{BACKEND_PORT}/v1/audio/speech",
            json={
                "input": text,
                "references": [{"audio_path": ref_file, "text": ref_text}],
            },
            timeout=60
        )
        elapsed = (time.time() - t_start) * 1000
        print(f"⏱ Toplam: {elapsed:.0f}ms | '{text[:40]}'")
        return {
            "audio_base64": base64.b64encode(response.content).decode(),
            "format": "wav",
            "sample_rate": 44100,
            "elapsed_ms": round(elapsed),
        }
    except Exception as e:
        return {"error": str(e)}


def start_sglang_backend():
    """
    SGLang-Omni'yi izole venv'deki Python ile başlat.
    Torchvision mismatch yok çünkü kendi ortamında.
    """
    print(f"[sglang] Python: {SGLANG_PYTHON}")
    print(f"[sglang] Config: {SGLANG_CONFIG}")
    print(f"[sglang] Model: {MODEL_PATH}")

    proc = subprocess.Popen(
        [
            SGLANG_PYTHON, "-m", "sglang_omni.cli.cli", "serve",
            "--model-path", MODEL_PATH,
            "--config", SGLANG_CONFIG,
            "--port", str(BACKEND_PORT),
            "--host", "0.0.0.0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    for line in proc.stdout:
        print(f"[sglang] {line.strip()}")


def wait_for_backend(timeout=300):
    print("⏳ SGLang-Omni bekleniyor...")
    for i in range(timeout // 10):
        time.sleep(10)
        try:
            r = requests.get(f"http://127.0.0.1:{BACKEND_PORT}/health", timeout=5)
            if r.status_code == 200:
                print(f"✅ SGLang-Omni hazır! ({(i+1)*10}sn)")
                return True
        except:
            if i % 3 == 0:
                print(f"⏳ {(i+1)*10}sn...")
    return False


if __name__ == "__main__":

    # Model indir (yoksa)
    if not os.path.exists(f"{MODEL_PATH}/codec.pth"):
        print("⏳ Model indiriliyor...")
        os.makedirs(MODEL_PATH, exist_ok=True)
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="fishaudio/s2-pro", local_dir=MODEL_PATH)
        print("✅ Model indirildi!")
    else:
        print("✅ Model mevcut!")

    # SGLang-Omni başlat (izole venv)
    print("🚀 SGLang-Omni başlatılıyor (izole venv)...")
    threading.Thread(target=start_sglang_backend, daemon=True).start()

    if not wait_for_backend():
        print("❌ SGLang-Omni başlamadı!")
        sys.exit(1)

    # OpenAI wrapper (port 8000)
    threading.Thread(target=run_openai_server, daemon=True).start()
    print("✅ OpenAI API hazır: port 8000")

    runpod.serverless.start({"handler": handler})
