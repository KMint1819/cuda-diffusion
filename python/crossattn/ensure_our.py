'''
We refactored the original CrossAttention from copied_crossattn.py to truncated_crossattn.py.
This code will verify that the outputs of the two implementations are the same.

TODO: WARNING:
The crossattn_our/CrossAttention somehow cannot pass this test even though the result of plugging
back into ControlNet is fine.
'''
from crossattn_truncated import CrossAttention as CrossAttention1
from crossattn_our import CrossAttention as CrossAttention2
import torch
import numpy as np
from pathlib import Path
torch.set_printoptions(sci_mode=False)
tolerance = 1e-4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(p, shape):
    raw = np.loadtxt(p, dtype=np.float32)
    tensor = torch.from_numpy(raw).reshape(shape)
    return tensor

kwargs = {
    'query_dim': 320,
    'context_dim': 768,
    'heads': 8,
    'dim_head': 40,
    'dropout': 0.0,
}
inner_dim = kwargs['dim_head'] * kwargs['heads'] 

data_dir = Path(__file__).cwd().parent.parent / 'data'
x = load_data(data_dir / 'input.txt', (1, 4096, 320))
context = load_data(data_dir / 'context.txt', (1, 77, 768))
state_dict = {
    'to_q.weight': load_data(data_dir / 'to_q-weight.txt', (inner_dim, kwargs['query_dim'])),
    'to_k.weight': load_data(data_dir / 'to_k-weight.txt', (inner_dim, kwargs['context_dim'])),
    'to_v.weight': load_data(data_dir / 'to_v-weight.txt', (inner_dim, kwargs['context_dim'])),
    'to_out.0.weight': load_data(data_dir / 'to_out-0-weight.txt', (kwargs['query_dim'], inner_dim)),
    'to_out.0.bias': load_data(data_dir / 'to_out-0-bias.txt', (kwargs['query_dim'],)),
}

attn1 = CrossAttention1(**kwargs)
attn2 = CrossAttention2(**kwargs)
print('=' * 80)
for k, v in attn1.state_dict().items():
    print(k, v.shape)
print('=' * 80)

attn1.load_state_dict(state_dict)
attn2.load_state_dict(state_dict)

attn1 = attn1.eval().to(device)
attn2 = attn2.eval().to(device)
x = x.to(device)
context = context.to(device)
with torch.no_grad():
    out1 = x.clone()
    out2 = x.clone()
    for i in range(10):
        out1 = attn1(out1, context)
        out2 = attn2(out2, context)

    print(f'Truncated output: ', out1)
    print(f'Ours      output: ', out2)
    # Compare two outputs
    
    if torch.allclose(out1, out2, atol=tolerance):
        print('Outputs are the same for the tolerance of ', tolerance)
    else:
        diffs = torch.argwhere((out1 - out2) > tolerance)
        print(f'Number of different elements: {diffs.shape}/{torch.numel(out1)}')
        print('diff shape: ', diffs.shape)
        print('out shape: ', out1.shape)
        for i, diff in enumerate(diffs[:10]):
            print(f'difference {i}: {out1[diff[0], diff[1], diff[2]]}, {out2[diff[0], diff[1], diff[2]]}')
        print('BAD!!!')