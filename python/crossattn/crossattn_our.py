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
        # return super().load_state_dict(state_dict, strict)

    # Mask was never specified in ControlNet
    def forward(self, x, context=None, mask=None):
        # print('x.shape: ', x.shape)
        context = default(context, x)
        return gten.compute(x, context)