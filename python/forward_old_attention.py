from attentionblock import AttentionBlock
import time
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

data_dir = Path(__file__).cwd().parent / 'data'
x = load_data(data_dir / 'input.txt', (1, n_channels, 32))
x = x.cuda()

state_dict ={
    'norm.weight': load_data(data_dir / 'norm-weight.txt', (n_channels,)),
    'norm.bias': load_data(data_dir / 'norm-bias.txt', (n_channels,)),
    'qkv.weight': load_data(data_dir / 'qkv-weight.txt', (n_channels * 3, n_channels, 1)),
    'qkv.bias': load_data(data_dir / 'qkv-bias.txt', (n_channels * 3,)),
    'proj_out.weight': load_data(data_dir / 'proj_out-weight.txt', (n_channels, n_channels, 1)),
    'proj_out.bias': load_data(data_dir / 'proj_out-bias.txt', (n_channels,))
}

block = AttentionBlock(
    channels = n_channels,
    num_heads = n_heads,
    num_head_channels = n_head_channels)
block.load_state_dict(state_dict)
block = block.cuda()

with torch.no_grad():
    start = time.time()
    out = block(x)
    end = time.time()
    print(f'Cost {end - start} seconds. After attention: {out.shape}\n{out}')