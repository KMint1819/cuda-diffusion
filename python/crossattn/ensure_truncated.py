'''
We refactored the original CrossAttention from copied_crossattn.py to truncated_crossattn.py.
This code will verify that the outputs of the two implementations are the same.
'''
from copied_crossattn import CrossAttention as CopiedCrossAttention
from truncated_crossattn import CrossAttention as TruncatedCrossAttention
import torch
import numpy as np
from pathlib import Path

def load_data(p, shape):
    raw = np.loadtxt(p, dtype=np.float32)
    tensor = torch.from_numpy(raw).reshape(shape)
    return tensor

kwargs = {
    'query_dim': 320,
    'context_dim': 320,
    'heads': 8,
    'dim_head': 40,
    'dropout': 0.0,
}

data_dir = Path(__file__).cwd().parent.parent / 'data'
x = load_data(data_dir / 'input.txt', (1, 4096, 320))
state_dict = {
    'to_q.weight': load_data(data_dir / 'to_q-weight.txt', (320, 320)),
    'to_k.weight': load_data(data_dir / 'to_k-weight.txt', (320, 320)),
    'to_v.weight': load_data(data_dir / 'to_v-weight.txt', (320, 320)),
    'to_out.0.weight': load_data(data_dir / 'to_out-0-weight.txt', (320, 320)),
    'to_out.0.bias': load_data(data_dir / 'to_out-0-bias.txt', (320,)),
}

copied = CopiedCrossAttention(**kwargs)
truncated = TruncatedCrossAttention(**kwargs)

copied.load_state_dict(state_dict)
truncated.load_state_dict(state_dict)

with torch.no_grad():
    copied_out = copied(x)
    truncated_out = truncated(x)

    print(f'Copied    output: ', copied_out)
    print(f'Truncated output: ', truncated_out)
    # Compare two outputs
    if torch.allclose(copied_out, truncated_out, atol=1e-6):
        print('Outputs are the same!')
    else:
        print('BAD!!!')