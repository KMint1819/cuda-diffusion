'''
Generate fake data for testing the C++ implementation of the attention block.

query_dim:  320
context_dim:  320
heads:  8
dim_head:  40
dropout:  0.0
inner_dim:  320
to_q.weight.shape:  torch.Size([320, 320])
to_k.weight.shape:  torch.Size([320, 320])
to_v.weight.shape:  torch.Size([320, 320])
to_out[0].weight.shape:  torch.Size([320, 320])
to_out[0].bias.shape:  torch.Size([320])

x.shape:  torch.Size([1, 4096, 320])
out.shape:  torch.Size([1, 4096, 320])
'''

from copied_crossattn import CrossAttention
import torch
import numpy as np

query_dim=  320
context_dim=  320
heads=  8
dim_head=  40
dropout=  0.0
inner_dim=  320

block = CrossAttention(
    query_dim = query_dim,
    context_dim = context_dim,
    heads = heads,
    dim_head = dim_head,
    dropout = dropout)

print('Model layers:', '=' * 80)
for k, v in block.state_dict().items():
    print(k, v.shape)
print('=' * 80)

# My fake input
input_data = torch.randn(1, 4096, 320)
np.savetxt('input.txt', input_data.numpy().reshape(-1).astype(np.float32), fmt='%.6f')

# Initialize the weights 
state_dict = {
    'to_q.weight': torch.randn((320, 320)),
    'to_k.weight': torch.randn((320, 320)),
    'to_v.weight': torch.randn((320, 320)),
    'to_out.0.weight': torch.randn((320, 320)),
    'to_out.0.bias': torch.randn(320),
}
block.load_state_dict(state_dict)
print('Model loaded state_dict successfully! Structure: \n', block)

# # My fake weights
for k, v in state_dict.items():
    name = k.replace('.', '-')
    np.savetxt(f'{name}.txt', v.numpy().reshape(-1), fmt='%.6f')

# # My fake output
with torch.no_grad():
    out = block(input_data)
    np.savetxt('out.txt', out.numpy().reshape(-1), fmt='%.6f')
    print('output.shape: ', out.shape)