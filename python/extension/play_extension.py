import torch
import torda
import numpy as np
from pathlib import Path
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

data_dir = Path(__file__).parent.parent.parent / 'data'
x = load_data(data_dir / 'input.txt', (1, n_channels, 32))
norm_weight = load_data(data_dir / 'norm-weight.txt', (n_channels,))
norm_bias = load_data(data_dir / 'norm-bias.txt', (n_channels,))
qkv_weight = load_data(data_dir / 'qkv-weight.txt', (n_channels * 3, n_channels, 1))
qkv_bias = load_data(data_dir / 'qkv-bias.txt', (n_channels * 3,))
proj_out_weight = load_data(data_dir / 'proj_out-weight.txt', (n_channels, n_channels, 1))
proj_out_bias = load_data(data_dir / 'proj_out-bias.txt', (n_channels,))

b, c, *spatial = x.shape
x = torda.preprocess(x)
print(f'After preprocess: {x.shape}')
print(f'After preprocess: {x}')

norm = torda.normalize(x, norm_weight, norm_bias, n_channels)
print(f'After normalize: {norm}')

qkv = torda.qkv(norm, qkv_weight, qkv_bias, n_channels, n_channels * 3, 1)
print(f'After qkv: {x}')

h = torda.attention(qkv, n_heads)
print(f'After attention: {x.shape}\n{x}')

h = torda.proj_out(h, proj_out_weight, proj_out_bias, n_channels, n_channels, 1)
print(f'After proj_out: {x.shape}\n{x}')

print(f'x: {x.shape}\n{x}')
print(f'h: {h.shape}\n{h}')
print(f'shape: {(b, c, *spatial)}')
x = torda.postprocess(x, h, (b, c, *spatial))
print(f'After postprocess: {x.shape}\n{x}')