'''
We refactored the original CrossAttention from copied_crossattn.py to truncated_crossattn.py.
This code will verify that the outputs of the two implementations are the same.
'''
from crossattn_truncated import CrossAttention as TruncatedCrossAttention
from crossattn_our import CrossAttention as OurCrossAttention
import torch
import numpy as np
from pathlib import Path

torch.set_printoptions(sci_mode=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

truncated = TruncatedCrossAttention(**kwargs)
our = OurCrossAttention(**kwargs)
print('=' * 80)
for k, v in truncated.state_dict().items():
    print(k, v.shape)
print('=' * 80)

truncated.load_state_dict(state_dict)
our.load_state_dict(state_dict)

with torch.no_grad():
    truncated = truncated.to(device)
    our = our.to(device)
    x = x.to(device)

    truncated_out = truncated(x).cpu()
    our_out = our(x).cpu()

    print(f'Truncated output: ', truncated_out)
    print(f'Our       output: ', our_out)
    # Compare two outputs
    if torch.allclose(truncated_out, our_out, atol=1e-4):
        print('Outputs are the same!')
    else:
        print('BAD!!!')