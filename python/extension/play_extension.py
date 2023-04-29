import torch
import torda
torch.set_printoptions(sci_mode=False)

n_channels = 64
n_heads = 8
n_head_channels = 32

x = torch.arange(0, n_channels * 32).reshape(1, n_channels, 32).float()
norm_weight= torch.ones(64) * 0.1
norm_bias = torch.ones(64) * 0.3
qkv_weight = torch.ones((192, 64, 1)) * 0.5
qkv_bias = torch.ones(192) * 0.7
proj_out_weight = torch.ones((64, 64, 1)) * 0.2
proj_out_bias = torch.ones(64) * 0.4

# print('x: ', x[:10])
# print('norm_weight: ', norm_weight[:10])
# print('norm_bias: ', norm_bias[:10])

x = torda.preprocess(x, norm_weight, norm_bias, n_channels)
print(f'After preprocess: {x.shape}')
print(f'After preprocess: {x}')

x = torda.qkv(x, qkv_weight, qkv_bias, n_channels, n_channels * 3, 1)
print(f'After qkv: {x}')

# x = torda.attention(x, 40)
# print(f'After attention: {x}')

# x = torda.proj_out(x,  torch.zeros((1, 2, 3)), torch.zeros((4, 5, 6)), 50, 60, 70)
# print(f'After proj_out: {x}')

# x = torda.postprocess(x, [100, 102, 104])
# print(f'After postprocess: {x}')