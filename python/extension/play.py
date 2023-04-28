import torch
import torda

x = torch.randn(2, 32, 32)
x = torda.preprocess(x)
print(f'After preprocess: {x}')

x = torda.normalize(x, 900)
print(f'After normalize: {x}')

x = torda.qkv(x, torch.zeros((1, 2, 3)), torch.zeros((4, 5, 6)), 10, 20, 30)
print(f'After qkv: {x}')

x = torda.attention(x, torch.zeros((1, 2, 3)), torch.zeros((4, 5, 6)), 40, )
print(f'After attention: {x}')

x = torda.proj_out(x,  torch.zeros((1, 2, 3)), torch.zeros((4, 5, 6)), 50, 60, 70)
print(f'After proj_out: {x}')

x = torda.postprocess(x, [100, 102, 104])
print(f'After postprocess: {x}')