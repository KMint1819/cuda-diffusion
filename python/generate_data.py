'''
Generate fake data for testing the C++ implementation of the attention block.
Input shape: (1, 64, 32)
Weights shape: {
    'norm.weight': (64),
    'norm.bias': (64),
    'qkv.weight': (192, 64, 1),
    'qkv.bias': (192),
    'proj_out.weight': (64, 64, 1),
    'proj_out.bias': (64),
}
Output shape: (1, 64, 32)
'''

from attentionblock_o import AttentionBlock
import torch
import numpy as np

n_channels = 64 # Must be a multiple of n_head_channels
n_heads = 8
n_head_channels = 32 # Must be a multiple of 32

block = AttentionBlock(
    channels = n_channels,
    num_heads = n_heads,
    num_head_channels = n_head_channels,
    use_checkpoint=True)
for k, v in block.state_dict().items():
    print(k, v.shape)

# My fake input
input_len = 32
input_data = torch.arange(0, 1 * n_channels * input_len) \
    .reshape(1, n_channels, input_len).float()
np.savetxt('input.txt', input_data.numpy().reshape(-1), fmt='%.6f')

# Initialize the weights 
state_dict = {
    'norm.weight': torch.ones(64) * 0.1,
    'norm.bias': torch.ones(64) * 0.3,
    'qkv.weight': torch.ones((192, 64, 1)) * 0.5,
    'qkv.bias': torch.ones(192) * 0.7,
    'proj_out.weight': torch.ones((64, 64, 1)) * 0.2,
    'proj_out.bias': torch.ones(64) * 0.4,
}

# My fake weights
for(k, v) in state_dict.items():
    name = k.replace('.', '-')
    np.savetxt(f'{name}.txt', v.numpy().reshape(-1), fmt='%.6f')

# My fake output
block.load_state_dict(state_dict)
with torch.no_grad():
    out = block(input_data)
    np.savetxt('out.txt', out.numpy().reshape(-1), fmt='%.6f')
    print(out.shape)