"""
Fish Speech S2-Pro + SGLang-Omni
Base image: frankleeeee/sglang-omni:dev
sglang_omni sistem Python'unda kurulu — path'i dinamik bul.
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


def find_sglang_python():
    """sglang_omni'nin kurulu olduğu Python'u bul"""
    candidates = [
        sys.executable,
        "/usr/bin/python3",
        "/usr/bin/python3.12",
        "/usr/local/bin/python3",
        "/usr/local/bin/python3.12",
        "/opt/conda/bin/python3",
        "/opt/conda/bin/python",
    ]
    for py in candidates:
        if not os.path.exists(py):
            continue
        result = subprocess.run(
            [py, "-c", "import sglang_omni; print(sglang_omni.__file__)"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"✅ sglang_omni: {py} → {result.stdout.strip()}")
            return py
    # Hiçbirinde bulunamazsa which ile ara
    result = subprocess.run(["find", "/", "-name", "sglang_omni", "-type", "d", "2>/dev/null"],
                            capture_output=True, text=True)
    print(f"⚠️ sglang_omni bulunamadı. sys.executable kullanılıyor: {sys.executable}")
    return sys.executable


def find_sglang_config():
    """s2pro_tts.yaml config dosyasını bul veya oluştur"""
    candidates = [
        "/tmp/sglang-omni/examples/configs/s2pro_tts.yaml",
        "/workspace/sglang-omni/examples/configs/s2pro_tts.yaml",
        "/app/sglang-omni/examples/configs/s2pro_tts.yaml",
    ]
    # Önce bul
    for path in candidates:
        if os.path.exists(path):
            print(f"✅ Config bulundu: {path}")
            return path

    # Yoksa oluştur
    config_path = "/tmp/s2pro_tts.yaml"
    with open(config_path, "w") as f:
        f.write(f"""model_path: {MODEL_PATH}
port: {BACKEND_PORT}
host: 0.0.0.0
dtype: float16
mem_fraction_static: 0.65
""")
    print(f"✅ Config oluşturuldu: {config_path}")
    return config_path


def start_sglang_backend():
    python_bin = find_sglang_python()
    config_path = find_sglang_config()

    print(f"[sglang] Python: {python_bin}")
    print(f"[sglang] Config: {config_path}")
    print(f"[sglang] Model: {MODEL_PATH}")

    proc = subprocess.Popen(
        [
            python_bin, "-m", "sglang_omni.cli.cli", "serve",
            "--model-path", MODEL_PATH,
            "--config", config_path,
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

    print("🚀 SGLang-Omni başlatılıyor...")
    threading.Thread(target=start_sglang_backend, daemon=True).start()

    if not wait_for_backend():
        print("❌ SGLang-Omni başlamadı!")
        sys.exit(1)

    threading.Thread(target=run_openai_server, daemon=True).start()
    print("✅ OpenAI API hazır: port 8000")

    runpod.serverless.start({"handler": handler})
