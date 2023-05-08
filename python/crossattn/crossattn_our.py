'''
Truncated CrossAttention
'''
import math
from typing import Any, Mapping
import torch
from torch import nn
from inspect import isfunction
from einops import rearrange, repeat
import os
import gten

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

        self.query_dim = query_dim
        self.context_dim = context_dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout

    def load_state_dict(self, state_dict, strict=True):
        gten.initialize(state_dict['to_q.weight'],
                                    state_dict['to_k.weight'],
                                    state_dict['to_v.weight'],
                                    state_dict['to_out.0.weight'],
                                    state_dict['to_out.0.bias'],
                                    self.query_dim,
                                    self.context_dim,
                                    self.heads,
                                    self.dim_head,
                                    self.dropout)
        return super().load_state_dict(state_dict, strict)

    def forward(self, x, context=None, mask=None):
        # print('x.shape: ', x.shape)
        context = default(context, x)
        out = gten.compute(x, context, mask)
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        with torch.autocast(enabled=False, device_type = 'cuda'):
            q, k = q.float(), k.float()
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