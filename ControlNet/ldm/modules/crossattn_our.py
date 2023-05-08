'''
Our CrossAttention
'''
import math
from typing import Any, Mapping
import torch
from torch import nn
from inspect import isfunction
from einops import rearrange, repeat
import os
import gten

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

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

    def load_state_dict(self, state_dict, strict=True):
        gten.initialize(self.to_q.weight, 
                        self.to_k.weight,
                        self.to_v.weight,
                        self.to_out[0].weight,
                        self.to_out[0].bias,
                        self.query_dim,
                        self.context_dim,
                        self.heads,
                        self.dim_head,
                        self.dropout)
        return super().load_state_dict(state_dict, strict)

    # Mask was never specified in ControlNet
    def forward(self, x, context=None, mask=None):
        # print('x.shape: ', x.shape)
        context = default(context, x)
        x = x.to(device)
        context = context.to(device)
        return gten.compute(x, context)