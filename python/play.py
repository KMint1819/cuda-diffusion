from attentionblock import AttentionBlock
import torch
from torch import nn
import numpy as np
from pathlib import Path
torch.set_printoptions(sci_mode=False)

def load_data(p, shape):
    raw = np.loadtxt(p, dtype=np.float32)
    tensor = torch.from_numpy(raw).reshape(shape)
    return tensor

n_channels = 64 # Must be a multiple of n_head_channels
n_heads = 8
n_head_channels = 32 # Must be a multiple of 32

data_dir = Path(__file__).parent.parent / 'data'
x = load_data(data_dir / 'input.txt', (1, n_channels, 32))
norm_weight = load_data(data_dir / 'norm-weight.txt', (n_channels,))
norm_bias = load_data(data_dir / 'norm-bias.txt', (n_channels,))
qkv_weight = load_data(data_dir / 'qkv-weight.txt', (n_channels * 3, n_channels, 1))
qkv_bias = load_data(data_dir / 'qkv-bias.txt', (n_channels * 3,))
proj_out_weight = load_data(data_dir / 'proj_out-weight.txt', (n_channels, n_channels, 1))
proj_out_bias = load_data(data_dir / 'proj_out-bias.txt', (n_channels,))

block = AttentionBlock(
    channels = n_channels,
    num_heads = n_heads,
    num_head_channels = n_head_channels)

block.norm.weight = nn.Parameter(norm_weight, requires_grad=False)
block.norm.bias = nn.Parameter(norm_bias, requires_grad=False)
block.qkv.weight = nn.Parameter(qkv_weight, requires_grad=False)
block.qkv.bias = nn.Parameter(qkv_bias, requires_grad=False)
block.proj_out.weight = nn.Parameter(proj_out_weight, requires_grad=False)
block.proj_out.bias = nn.Parameter(proj_out_bias, requires_grad=False)

with torch.no_grad():
    out = block(x)
    print(f'After attention: {out.shape}\n{out}')