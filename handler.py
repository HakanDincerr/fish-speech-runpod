"""
Fish Speech S2-Pro + SGLang-Omni handler
sglang_omni hangi python'da kuruluysa onu bul ve kullan
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

sys.path.insert(0, "/app/fish-speech")

_REF_AUDIO_PATH = "/app/referans.mp3"
REF_AUDIO_FILE = "/app/referans.mp3"

if os.path.exists(_REF_AUDIO_PATH):
    with open(_REF_AUDIO_PATH, "rb") as f:
        REF_AUDIO_B64 = base64.b64encode(f.read()).decode()
else:
    REF_AUDIO_B64 = os.environ.get("REF_AUDIO_B64", "")
    if REF_AUDIO_B64:
        with open(REF_AUDIO_FILE, "wb") as f:
            f.write(base64.b64decode(REF_AUDIO_B64))

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
    ref_text = body.get("ref_text", REF_TEXT)

    ref_audio_b64 = body.get("ref_audio", None)
    if ref_audio_b64:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.write(base64.b64decode(ref_audio_b64))
        tmp.close()
        ref_audio_file = tmp.name
    else:
        ref_audio_file = REF_AUDIO_FILE

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
                    "references": [{"audio_path": ref_audio_file, "text": ref_text}],
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
        ref_audio_file = tmp.name
    else:
        ref_audio_file = REF_AUDIO_FILE

    t_start = time.time()
    try:
        response = requests.post(
            f"http://127.0.0.1:{BACKEND_PORT}/v1/audio/speech",
            json={
                "input": text,
                "references": [{"audio_path": ref_audio_file, "text": ref_text}],
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


def find_python_with_sglang():
    """sglang_omni'nin kurulu olduğu python'u bul"""
    candidates = [
        "/usr/local/bin/python3",
        "/usr/local/bin/python",
        "/usr/bin/python3.11",
        "/usr/bin/python3",
        sys.executable,
    ]
    for py in candidates:
        if not os.path.exists(py):
            continue
        result = subprocess.run(
            [py, "-c", "import sglang_omni; print(sglang_omni.__file__)"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"✅ sglang_omni bulundu: {py} → {result.stdout.strip()}")
            return py
    # Bulamazsak mevcut python'u dene
    print(f"⚠️ sglang_omni bulunamadı, {sys.executable} deneniyor")
    return sys.executable


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
    python_bin = find_python_with_sglang()

    config_candidates = [
        "/tmp/sglang-omni/examples/configs/s2pro_tts.yaml",
        "/app/s2pro_tts.yaml",
    ]
    config_path = next((p for p in config_candidates if os.path.exists(p)), "/app/s2pro_tts.yaml")

    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            f.write(f"""model_path: {MODEL_PATH}
port: {BACKEND_PORT}
host: 0.0.0.0
dtype: float16
mem_fraction_static: 0.65
""")

    print(f"[sglang] Python: {python_bin}")
    print(f"[sglang] Config: {config_path}")

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

    fix_torchaudio()

    if not os.path.exists(f"{MODEL_PATH}/codec.pth"):
        print("⏳ Model indiriliyor...")
        os.makedirs(MODEL_PATH, exist_ok=True)
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="fishaudio/s2-pro", local_dir=MODEL_PATH)
        print("✅ Model indirildi!")
    else:
        print("✅ Model mevcut!")

    print("🚀 SGLang-Omni S2Pro başlatılıyor...")
    threading.Thread(target=start_sglang_backend, daemon=True).start()

    if not wait_for_backend():
        print("❌ SGLang-Omni başlamadı!")
        sys.exit(1)

    threading.Thread(target=run_openai_server, daemon=True).start()
    print("✅ OpenAI API hazır: port 8000")

    runpod.serverless.start({"handler": handler})
