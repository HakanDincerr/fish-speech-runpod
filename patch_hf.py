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

# 2. torchvision extension stub
tv_ext = f'{site}/torchvision/extension.py'
if os.path.exists(tv_ext):
    with open(tv_ext, 'w') as f:
        f.write('def _has_ops():\n    return False\n\n_HAS_OPS = False\n')
    print('2. torchvision/extension.py stubbed')

# 3. torchvision _meta_registrations stub
tv_meta = f'{site}/torchvision/_meta_registrations.py'
if os.path.exists(tv_meta):
    with open(tv_meta, 'w') as f:
        f.write('# stub - all meta registrations disabled\n')
    print('3. torchvision/_meta_registrations.py stubbed')

# 4. torchvision __init__ - sadece extension ve transforms import et
tv_init = f'{site}/torchvision/__init__.py'
if os.path.exists(tv_init):
    with open(tv_init, 'w') as f:
        f.write('''from torchvision.extension import _HAS_OPS
from torchvision import transforms

class _Stub:
    def __getattr__(self, name):
        return self
    def __call__(self, *a, **kw):
        return self
    def __bool__(self):
        return False

datasets = _Stub()
io = _Stub()
models = _Stub()
ops = _Stub()
utils = _Stub()
''')
    print('4. torchvision/__init__.py stubbed')

# 5. torchvision transforms stub
tv_transforms = f'{site}/torchvision/transforms/__init__.py'
if os.path.exists(tv_transforms):
    with open(tv_transforms, 'w') as f:
        f.write('''class InterpolationMode:
    BICUBIC = "bicubic"
    BILINEAR = "bilinear"
    NEAREST = "nearest"
    LANCZOS = "lanczos"
    NEAREST_EXACT = "nearest_exact"

class AutoAugmentPolicy:
    CIFAR10 = "cifar10"
    IMAGENET = "imagenet"
    SVHN = "svhn"

class functional:
    @staticmethod
    def to_tensor(x): return x
    @staticmethod
    def normalize(x, *a, **kw): return x

class Compose:
    def __init__(self, t): self.t = t
    def __call__(self, x):
        for t in self.t: x = t(x)
        return x

class ToTensor:
    def __call__(self, x): return x

class Normalize:
    def __init__(self, *a, **kw): pass
    def __call__(self, x): return x

class Resize:
    def __init__(self, *a, **kw): pass
    def __call__(self, x): return x

class CenterCrop:
    def __init__(self, *a, **kw): pass
    def __call__(self, x): return x
''')
    print('5. torchvision/transforms/__init__.py stubbed')

# 6. torchvision transforms v2 stub
tv_v2 = f'{site}/torchvision/transforms/v2'
if os.path.exists(tv_v2):
    init_v2 = f'{tv_v2}/__init__.py'
    with open(init_v2, 'w') as f:
        f.write('from torchvision.transforms import AutoAugmentPolicy, InterpolationMode, functional\n')
    print('6. torchvision/transforms/v2 stubbed')

# 7. torchvision transforms functional stub
tv_func = f'{site}/torchvision/transforms/functional.py'
if os.path.exists(tv_func):
    with open(tv_func, 'w') as f:
        f.write('''from torchvision.transforms import InterpolationMode
def to_tensor(x): return x
def normalize(x, *a, **kw): return x
def resize(x, *a, **kw): return x
def center_crop(x, *a, **kw): return x
def pil_to_tensor(x): return x
''')
    print('7. torchvision/transforms/functional.py stubbed')

# 8. transformers import_utils - is_torchvision_available False yap
import_utils = f'{site}/transformers/utils/import_utils.py'
if os.path.exists(import_utils):
    with open(import_utils) as f:
        content = f.read()
    with open(import_utils, 'w') as f:
        f.write(content + '\n\ndef is_torchvision_available():\n    return False\n')
    print('8. transformers is_torchvision_available patched')

print('All patches done!')
