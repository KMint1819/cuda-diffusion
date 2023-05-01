import torch
import torda
from pathlib import Path
from collections import OrderedDict
import numpy as np

def load_data(p, shape):
    raw = np.loadtxt(p, dtype=np.float32)
    tensor = torch.from_numpy(raw).reshape(shape)
    return tensor

n_channels = 64
n_heads = 8
n_head_channels = 32

data_dir = Path(__file__).cwd().parent.parent / 'data'
print(data_dir.resolve())
state_dict = OrderedDict({
    'norm.weight': load_data(data_dir / 'norm-weight.txt', (n_channels,)),
    'norm.bias': load_data(data_dir / 'norm-bias.txt', (n_channels,)),
    'qkv.weight': load_data(data_dir / 'qkv-weight.txt', (n_channels * 3, n_channels, 1)),
    'qkv.bias': load_data(data_dir / 'qkv-bias.txt', (n_channels * 3,)),
    'proj_out.weight': load_data(data_dir / 'proj_out-weight.txt', (n_channels, n_channels, 1)),
    'proj_out.bias': load_data(data_dir / 'proj_out-bias.txt', (n_channels,))
})

print(state_dict.keys())
args = []
for k, v in state_dict.items():
    args.append(v)
torda.initialize(*args)