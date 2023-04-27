from attentionblock_o import AttentionBlock as OldAttentionBlock
from attentionblock import AttentionBlock as NewAttentionBlock
import torch
import numpy as np

n_channels = 64 # Must be a multiple of n_head_channels
n_heads = 8
n_head_channels = 32 # Must be a multiple of 32

oldBlock = OldAttentionBlock(
    channels = n_channels,
    num_heads = n_heads,
    num_head_channels = n_head_channels,
    use_checkpoint=True)

newBlock = NewAttentionBlock(
    channels = n_channels,
    num_heads = n_heads,
    num_head_channels = n_head_channels
)

for k, v in oldBlock.state_dict().items():
    print(k, v.shape)

# My fake input
input_len = 32
input_data = np.loadtxt('data/input.txt', dtype=np.float32)
input_data = torch.from_numpy(input_data).reshape(1, n_channels, input_len).float()

state_dict = {}
for k, v in oldBlock.state_dict().items():
    name = k.replace('.', '-')
    state_dict[k] = torch.from_numpy(np.loadtxt(f'data/{name}.txt', dtype=np.float32)).reshape(v.shape)

oldBlock.load_state_dict(state_dict)
newBlock.load_state_dict(state_dict)

with torch.no_grad():
    old_out = oldBlock(input_data)
    new_out = newBlock(input_data)

    # Compare two outputs
    print(torch.allclose(old_out, new_out, atol=1e-6))