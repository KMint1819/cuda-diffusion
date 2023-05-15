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
from gten_backend import GtenCrossAttention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# TODO: Make whole thing in C++
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        context_dim = default(context_dim, query_dim)
        inner_dim = dim_head * heads
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.is_backend_initialized = False

        # Although not used, these are required to successfully read data from model.load_state_dict 
        # Will be deleted after the backend is initialized
        self.to_q = nn.ParameterDict({
            'weight': nn.Parameter(torch.Tensor(inner_dim, query_dim))
        })
        self.to_k = nn.ParameterDict({
            'weight': nn.Parameter(torch.Tensor(inner_dim, context_dim))
        })
        self.to_v = nn.ParameterDict({
            'weight': nn.Parameter(torch.Tensor(inner_dim, context_dim))
        })
        self.to_out = nn.Sequential(
            nn.ParameterDict({
                'weight': nn.Parameter(torch.Tensor(query_dim, inner_dim)),
                'bias': nn.Parameter(torch.Tensor(query_dim))
            })
        )
        # print('\n\n', "#" * 80)
        # print("query_dim: ", query_dim)
        # print("context_dim: ", context_dim)
        # print("heads: ", heads)
        # print("dim_head: ", dim_head)
        # print("dropout: ", dropout)
        # print("inner_dim: ", inner_dim)
        # print("#" * 80)
        self.backend = GtenCrossAttention(self.query_dim, self.context_dim, self.heads, self.dim_head, self.dropout)

    def to(self, device):
        super().to(device)
        self.backend.to(device)
        return self

    # Mask was never specified in ControlNet
    def forward(self, x, context=None, mask=None):
        context = default(context, x)
        if not self.is_backend_initialized:
            self.backend.initialize(self.to_q.weight, self.to_k.weight, self.to_v.weight, self.to_out[0].weight, self.to_out[0].bias)
            self.is_backend_initialized = True
            del self.to_q
            del self.to_k
            del self.to_v
            del self.to_out
        out = self.backend.compute(x, context)
        torch.cuda.synchronize()
        return out