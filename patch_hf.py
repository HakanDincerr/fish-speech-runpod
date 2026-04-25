import re

hf_path = '/usr/local/lib/python3.11/dist-packages/sglang_omni/utils/hf.py'

new_content = '''from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

@contextmanager
def no_init_weights(_enable=True):
    yield

def load_pretrained_model_state_dict(model, state_dict, strict=True):
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")
    return model
'''

with open(hf_path, 'w') as f:
    f.write(new_content)

print('hf.py patched successfully!')
