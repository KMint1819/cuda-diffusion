'''
Copied CrossAttention
'''
import math
import torch
from torch import nn
from inspect import isfunction
from einops import rearrange, repeat
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        # print('=' * 80)
        # print('query_dim: ', query_dim)
        # print('context_dim: ', context_dim)
        # print('heads: ', heads)
        # print('dim_head: ', dim_head)
        # print('dropout: ', dropout)
        # print('inner_dim: ', inner_dim)
        # print('to_q.weight.shape: ', self.to_q.weight.shape)
        # print('to_k.weight.shape: ', self.to_k.weight.shape)
        # print('to_v.weight.shape: ', self.to_v.weight.shape)
        # print('to_out[0].weight.shape: ', self.to_out[0].weight.shape)
        # print('to_out[0].bias.shape: ', self.to_out[0].bias.shape)

    def forward(self, x, context=None, mask=None):
        # print('x.shape: ', x.shape)
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
    
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out = self.to_out(out)
        # print('out.shape: ', out.shape)
        return out