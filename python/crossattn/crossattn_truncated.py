'''
Truncated CrossAttention
'''
import math
import torch
from torch import nn
from inspect import isfunction
from einops import rearrange, repeat
import os

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
        print('\n', '=' * 80)
        print('query_dim: ', query_dim)
        print('context_dim: ', context_dim)
        print('heads: ', heads)
        print('dim_head: ', dim_head)
        print('dropout: ', dropout)
        print('inner_dim: ', inner_dim)

    def rearrange(self, tensor, h):
        b, n = tensor.shape[:2]
        d = tensor.shape[-1] // h
        tensor = tensor.reshape(b, n, h, d)
        tensor = tensor.permute(0, 2, 1, 3)
        tensor = tensor.reshape(b * h, n, d)
        return tensor

    def forward(self, x, context=None, mask=None):
        context = default(context, x)
        print('\n', '#' * 80)
        print('x.shape: ', x.shape)
        print('context.shape: ', context.shape)
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        # print('q: ', q)
        # print('k: ', k)
        # print('v: ', v)
        # print('q.shape: ', q.shape)
        # print('k.shape: ', k.shape)
        # print('v.shape: ', v.shape)
        # print('h: ', h)

        # TODO: make this look better
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        b = q.shape[0]
        n = q.shape[1]
        d = q.shape[2] // h
        q = self.rearrange(q, h)
        k = self.rearrange(k, h)
        v = self.rearrange(v, h)

        # print('q: ', q)
        # print('k: ', k)
        # print('v: ', v)
        # print('q.shape: ', q.shape)
        # print('k.shape: ', k.shape)
        # print('v.shape: ', v.shape)

        # force cast to fp32 to avoid overflowing
        with torch.autocast(enabled=False, device_type = 'cuda'):
            q, k = q.float(), k.float()
            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
    
        # The mask was never specified in the ControlNet
        # if exists(mask):
        #     print('mask.shape')
        #     mask = rearrange(mask, 'b ... -> b (...)')
        #     max_neg_value = -torch.finfo(sim.dtype).max
        #     mask = repeat(mask, 'b j -> (b h) () j', h=h)
        #     sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', sim, v)
        # change to normal operations
        # out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        
        out = out.reshape(b, h, n, d)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(b, n, h * d)

        out = self.to_out(out)
        # print('out.shape: ', out.shape)
        return out