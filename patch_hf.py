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

# 2. torchvision transforms stub - tam liste
tv_transforms = f'{site}/torchvision/transforms/__init__.py'
if os.path.exists(tv_transforms):
    with open(tv_transforms, 'w') as f:
        f.write('''# torchvision transforms stub
class InterpolationMode:
    BICUBIC = "bicubic"
    BILINEAR = "bilinear"
    NEAREST = "nearest"
    LANCZOS = "lanczos"
    NEAREST_EXACT = "nearest_exact"

class AutoAugmentPolicy:
    CIFAR10 = "cifar10"
    IMAGENET = "imagenet"
    SVHN = "svhn"

class AutoAugment:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, x): return x

class RandAugment:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, x): return x

class TrivialAugmentWide:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, x): return x

class AugMix:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, x): return x

class Compose:
    def __init__(self, transforms): self.transforms = transforms
    def __call__(self, x):
        for t in self.transforms: x = t(x)
        return x

class ToTensor:
    def __call__(self, x): return x

class Normalize:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, x): return x

class Resize:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, x): return x

class CenterCrop:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, x): return x

class RandomCrop:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, x): return x

class RandomHorizontalFlip:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, x): return x

class RandomVerticalFlip:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, x): return x

def pil_to_tensor(x): return x

class functional:
    @staticmethod
    def pil_to_tensor(x): return x
    @staticmethod
    def to_tensor(x): return x
    @staticmethod
    def normalize(x, *args, **kwargs): return x
''')
    print('2. torchvision.transforms stubbed')

# 3. torchvision __init__ stub
tv_init = f'{site}/torchvision/__init__.py'
if os.path.exists(tv_init):
    with open(tv_init, 'w') as f:
        f.write('''# torchvision stub
from torchvision import transforms

class _Stub:
    def __getattr__(self, name):
        return _Stub()
    def __call__(self, *args, **kwargs):
        return _Stub()

datasets = _Stub()
io = _Stub()
models = _Stub()
ops = _Stub()
utils = _Stub()

def __getattr__(name):
    return _Stub()
''')
    print('3. torchvision.__init__ stubbed')

# 4. transformers image_utils - torchvision import devre disi
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
