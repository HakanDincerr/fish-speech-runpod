import os
import re

site = '/usr/local/lib/python3.11/dist-packages'

# 1. hf.py patch
hf_path = f'{site}/sglang_omni/utils/hf.py'
with open(hf_path) as f:
    content = f.read()
stub = (
    'from contextlib import contextmanager\n\n'
    '@contextmanager\n'
    'def no_init_weights(_enable=True):\n'
    '    yield\n\n'
)
patched = re.sub(
    r'try:\s*\n\s*from transformers[^\n]*no_init_weights[^\n]*\n.*?except ImportError:\s*\n\s*from transformers[^\n]*no_init_weights',
    stub.strip(),
    content,
    flags=re.DOTALL
)
with open(hf_path, 'w') as f:
    f.write(patched)
print('1. hf.py patched')

# 2. transformers import_utils.py - is_torchvision_available her zaman False dondurecek
import_utils = f'{site}/transformers/utils/import_utils.py'
if os.path.exists(import_utils):
    with open(import_utils) as f:
        content = f.read()
    # is_torchvision_available fonksiyonunu bul ve override et
    patched = content + '\n\n# PATCH: disable torchvision\ndef is_torchvision_available():\n    return False\n'
    with open(import_utils, 'w') as f:
        f.write(patched)
    print('2. transformers/utils/import_utils.py patched - torchvision disabled')
else:
    print('2. import_utils.py not found')

print('All patches done!')
