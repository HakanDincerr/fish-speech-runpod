#!/bin/bash
python3 << 'EOF'
with open('/app/fish-speech/fish_speech/inference_engine/reference_loader.py', 'r') as f:
    content = f.read()

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

with open('/app/fish-speech/fish_speech/inference_engine/reference_loader.py', 'w') as f:
    f.write(content.replace(old, new))

print('✅ torchaudio fix applied')
EOF
