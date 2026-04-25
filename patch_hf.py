import os

# 1. hf.py patch - no_init_weights
hf_path = '/usr/local/lib/python3.11/dist-packages/sglang_omni/utils/hf.py'
with open(hf_path) as f:
    content = f.read()

import re
stub = ('from contextlib import contextmanager\n\n'
        '@contextmanager\n'
        'def no_init_weights(_enable=True):\n'
        '    yield\n\n')

patched = re.sub(
    r'try:\s*\n\s*from transformers[^\n]*no_init_weights[^\n]*\n.*?except ImportError:\s*\n\s*from transformers[^\n]*no_init_weights',
    stub.strip(),
    content,
    flags=re.DOTALL
)

with open(hf_path, 'w') as f:
    f.write(patched)
print('hf.py patched!')

# 2. torchvision _meta_registrations.py patch
# register_fake cagrisini devre disi birak
tv_meta_path = '/usr/local/lib/python3.11/dist-packages/torchvision/_meta_registrations.py'
if os.path.exists(tv_meta_path):
    with open(tv_meta_path) as f:
        tv_content = f.read()
    # register_fake cagrilarini no-op yap
    tv_patched = tv_content.replace(
        '@torch.library.register_fake("torchvision::nms")',
        '# @torch.library.register_fake("torchvision::nms")\nif False:'
    ).replace(
        '@torch.library.register_fake("torchvision::roi_align")',
        '# @torch.library.register_fake("torchvision::roi_align")\nif False:'
    ).replace(
        '@torch.library.register_fake("torchvision::roi_pool")',
        '# @torch.library.register_fake("torchvision::roi_pool")\nif False:'
    ).replace(
        '@torch.library.register_fake("torchvision::_new_empty_tensor_op")',
        '# @torch.library.register_fake("torchvision::_new_empty_tensor_op")\nif False:'
    )
    with open(tv_meta_path, 'w') as f:
        f.write(tv_patched)
    print('torchvision _meta_registrations.py patched!')
else:
    print('torchvision _meta_registrations.py not found, skipping')

print('All patches applied!')
