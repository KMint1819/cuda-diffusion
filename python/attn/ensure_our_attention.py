'''
Ensure our C++ implementation of the attention block is correct.
'''
from attentionblock import AttentionBlock
from our_attentionblock import AttentionBlock as OurAttentionBlock
import torch
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

ourBlock= OurAttentionBlock(
    channels = n_channels,
    num_heads = n_heads,
    num_head_channels = n_head_channels)

for k, v in block.state_dict().items():
    print(k, v.shape)

state_dict = {
    'norm.weight': norm_weight,
    'norm.bias': norm_bias,
    'qkv.weight': qkv_weight,
    'qkv.bias': qkv_bias,
    'proj_out.weight': proj_out_weight,
    'proj_out.bias': proj_out_bias
}
block.load_state_dict(state_dict)
ourBlock.load_state_dict(state_dict)

x = x.cuda()
block = block.cuda()
ourBlock = ourBlock.cuda()

with torch.no_grad():
    out = block(x)
    our_out = ourBlock(x)

    print(f'Original output: ', out)
    print(f'Out      output: ', our_out)
    # Compare two outputs
    if torch.allclose(out, our_out, atol=1e-6):
        print('Outputs are the same!')
    else:
        print('BAD!!!')