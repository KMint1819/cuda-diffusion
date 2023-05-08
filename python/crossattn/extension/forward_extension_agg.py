import torch
import torda
import numpy as np
from pathlib import Path
import time

torch.set_printoptions(sci_mode=False)

def load_data(p, shape):
    raw = np.loadtxt(p, dtype=np.float32)
    tensor = torch.from_numpy(raw).reshape(shape)
    return tensor

n_channels = 64
n_heads = 8
n_head_channels = 32

if n_head_channels != -1:
    assert (
        n_channels % n_head_channels == 0
    ), f"q,k,v channels {n_channels} is not divisible by num_head_channels {n_head_channels}"
    n_heads = n_channels // n_head_channels

data_dir = Path(__file__).cwd().parent.parent / 'data'
x = load_data(data_dir / 'input.txt', (1, n_channels, 32)).cuda()

state_dict = {
    'norm.weight': load_data(data_dir / 'norm-weight.txt', (n_channels,)).cuda(),
    'norm.bias': load_data(data_dir / 'norm-bias.txt', (n_channels,)).cuda(),
    'qkv.weight': load_data(data_dir / 'qkv-weight.txt', (n_channels * 3, n_channels, 1)).cuda(),
    'qkv.bias': load_data(data_dir / 'qkv-bias.txt', (n_channels * 3,)).cuda(),
    'proj_out.weight': load_data(data_dir / 'proj_out-weight.txt', (n_channels, n_channels, 1)).cuda(),
    'proj_out.bias': load_data(data_dir / 'proj_out-bias.txt', (n_channels,)).cuda()
}

# torda.func(state_dict)
torda.initialize(
    state_dict['norm.weight'], 
    state_dict['norm.bias'], 
    state_dict['qkv.weight'], 
    state_dict['qkv.bias'], 
    state_dict['proj_out.weight'], 
    state_dict['proj_out.bias'],
    n_channels,
    n_heads,)
start = time.time()
out = torda.compute(x, n_channels, n_heads)
end = time.time()
print(f'Cost {end - start} seconds. After compute: {out.shape}\n{out}')