import re

path = '/usr/local/lib/python3.11/dist-packages/sglang_omni/utils/hf.py'

with open(path) as f:
    content = f.read()

stub = (
    'from contextlib import contextmanager\n\n'
    '@contextmanager\n'
    'def no_init_weights(_enable=True):\n'
    '    yield\n\n'
)

# Sadece try/except import bloğunu değiştir, diğer fonksiyonlar kalır
patched = re.sub(
    r'try:\s*\n\s*from transformers[^\n]*no_init_weights[^\n]*\n[^\n]*\n\s*from transformers[^\n]*no_init_weights[^\n]*',
    stub,
    content,
    flags=re.DOTALL
)

if patched == content:
    # Farklı format dene
    patched = re.sub(
        r'try:.*?from transformers.*?no_init_weights.*?except ImportError:.*?from transformers.*?no_init_weights',
        stub.strip(),
        content,
        flags=re.DOTALL
    )

with open(path, 'w') as f:
    f.write(patched)

print('hf.py patched successfully!')
print('First 20 lines:')
with open(path) as f:
    for i, line in enumerate(f.readlines()[:20], 1):
        print(f'{i}: {line}', end='')
