'''
We refactored the original AttentionBlock from attentionblock_o.py to attentionblock.py.
This code will verify that the outputs of the two implementations are the same.
'''
from attentionblock_o import AttentionBlock as OldAttentionBlock
from attentionblock import AttentionBlock as NewAttentionBlock
import torch
import numpy as np
from pathlib import Path

def load_data(p, shape):
    raw = np.loadtxt(p, dtype=np.float32)
    tensor = torch.from_numpy(raw).reshape(shape)
    return tensor

n_channels = 64 # Must be a multiple of n_head_channels
n_heads = 8
n_head_channels = 32 # Must be a multiple of 32

data_dir = Path('../data')
x = load_data(data_dir / 'input.txt', (1, n_channels, 32))
norm_weight = load_data(data_dir / 'norm-weight.txt', (n_channels,))
norm_bias = load_data(data_dir / 'norm-bias.txt', (n_channels,))
qkv_weight = load_data(data_dir / 'qkv-weight.txt', (n_channels * 3, n_channels, 1))
qkv_bias = load_data(data_dir / 'qkv-bias.txt', (n_channels * 3,))
proj_out_weight = load_data(data_dir / 'proj_out-weight.txt', (n_channels, n_channels, 1))
proj_out_bias = load_data(data_dir / 'proj_out-bias.txt', (n_channels,))

oldBlock = OldAttentionBlock(
    channels = n_channels,
    num_heads = n_heads,
    num_head_channels = n_head_channels,
    use_checkpoint=True)

newBlock = NewAttentionBlock(
    channels = n_channels,
    num_heads = n_heads,
    num_head_channels = n_head_channels)

for k, v in oldBlock.state_dict().items():
    print(k, v.shape)

state_dict = {
    'norm.weight': norm_weight,
    'norm.bias': norm_bias,
    'qkv.weight': qkv_weight,
    'qkv.bias': qkv_bias,
    'proj_out.weight': proj_out_weight,
    'proj_out.bias': proj_out_bias
}

oldBlock.load_state_dict(state_dict)
newBlock.load_state_dict(state_dict)

with torch.no_grad():
    old_out = oldBlock(x)
    new_out = newBlock(x)

    # Compare two outputs
    print(torch.allclose(old_out, new_out, atol=1e-6))