import os
import re
import sys

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

# 2. torchvision'i uninstall et, stub koy
tv_init = f'{site}/torchvision/__init__.py'
if os.path.exists(tv_init):
    stub_tv = '''# torchvision stub - binary uyumsuzlugu nedeniyle devre disi
class _Stub:
    def __getattr__(self, name):
        raise ImportError(f"torchvision.{name} not available (stub mode)")

datasets = _Stub()
io = _Stub()
models = _Stub()
ops = _Stub()
transforms = _Stub()
utils = _Stub()

class _HAS_OPS:
    pass

def __getattr__(name):
    raise ImportError(f"torchvision.{name} not available (stub mode)")
'''
    with open(tv_init, 'w') as f:
        f.write(stub_tv)
    print('2. torchvision stubbed')

# 3. torchvision.transforms stub
tv_transforms = f'{site}/torchvision/transforms/__init__.py'
if os.path.exists(tv_transforms):
    with open(tv_transforms, 'w') as f:
        f.write('class InterpolationMode:\n    BICUBIC = "bicubic"\n    BILINEAR = "bilinear"\n    NEAREST = "nearest"\n\ndef functional():\n    pass\n')
    print('3. torchvision.transforms stubbed')

# 4. transformers image_utils - torchvision import'u bos gecir
img_utils = f'{site}/transformers/image_utils.py'
if os.path.exists(img_utils):
    with open(img_utils) as f:
        content = f.read()
    patched = content.replace(
        'if is_torchvision_available():',
        'if False:  # torchvision disabled'
    )
    with open(img_utils, 'w') as f:
        f.write(patched)
    print('4. transformers/image_utils.py patched')

print('All patches done!')
