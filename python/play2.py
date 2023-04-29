from attentionblock import AttentionBlock
import torch
from torch import nn
import numpy as np
torch.set_printoptions(sci_mode=False)

n_channels = 64 # Must be a multiple of n_head_channels
n_heads = 8
n_head_channels = 32 # Must be a multiple of 32

x = torch.arange(0, n_channels * 32).reshape(1, n_channels, 32).float()
norm_weight= torch.ones(64) * 0.1
norm_bias = torch.ones(64) * 0.3

block = AttentionBlock(
    channels = n_channels,
    num_heads = n_heads,
    num_head_channels = n_head_channels)

block.norm.weight = nn.Parameter(norm_weight)
block.norm.bias = nn.Parameter(norm_bias)

with torch.no_grad():
    out = block(x)
    print(out)
    # print(out.shape)