"""
Fish Speech S2-Pro + SGLang-Omni handler
Base: runpod/pytorch:2.4.0 + sgl-project-dev/sglang-omni
Hedef: ~300-500ms TTFB, H100 SXM
"""
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

    t_start = time.time()
    first_chunk = True

    async def generate():
        nonlocal first_chunk
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                f"http://127.0.0.1:{BACKEND_PORT}/v1/tts",
                json={
                    "text": text,
                    "references": [{"audio": ref_audio, "text": ref_text}],
                    "format": "wav",
                    "streaming": True,
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
    ref_audio = job_input.get("ref_audio", REF_AUDIO_B64)
    ref_text = job_input.get("ref_text", REF_TEXT)

    if not text:
        return {"error": "text is required"}

    t_start = time.time()
    try:
        response = requests.post(
            f"http://127.0.0.1:{BACKEND_PORT}/v1/tts",
            json={
                "text": text,
                "references": [{"audio": ref_audio, "text": ref_text}],
                "format": "wav",
                "streaming": False,
            },
            timeout=60
        )
        elapsed = (time.time() - t_start) * 1000
        print(f"⏱ Toplam: {elapsed:.0f}ms")
        return {
            "audio_base64": base64.b64encode(response.content).decode(),
            "format": "wav",
            "sample_rate": 44100
        }
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
            print("✅ torchaudio fix!")
    except Exception as e:
        print(f"⚠️ fix: {e}")


def start_sglang_backend():
    """SGLang-Omni S2Pro server"""
    config_path = "/app/s2pro_tts.yaml"
    with open(config_path, "w") as f:
        f.write(f"""model_config:
  model_type: s2pro
  model_path: {MODEL_PATH}
  device: cuda
  dtype: float16

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

    fix_torchaudio()

    # Model indir (yoksa)
    if not os.path.exists(f"{MODEL_PATH}/codec.pth"):
        print("⏳ Model indiriliyor...")
        os.makedirs(MODEL_PATH, exist_ok=True)
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="fishaudio/s2-pro", local_dir=MODEL_PATH)
        print("✅ Model indirildi!")
    else:
        print("✅ Model mevcut!")

    # SGLang-Omni başlat
    print("🚀 SGLang-Omni S2Pro başlatılıyor...")
    threading.Thread(target=start_sglang_backend, daemon=True).start()

    if not wait_for_backend():
        print("❌ SGLang-Omni başlamadı! Log:")
        sys.exit(1)

    # OpenAI wrapper
    threading.Thread(target=run_openai_server, daemon=True).start()
    print("✅ OpenAI API hazır: port 8000")

    # RunPod handler
    runpod.serverless.start({"handler": handler})
