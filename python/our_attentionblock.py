from typing import Any, Mapping
import torch
from torch import nn
import torda

class OurAttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        
        self.norm = nn.ParameterDict({
            'weight': nn.Parameter(torch.empty(channels), requires_grad=False),
            'bias': nn.Parameter(torch.empty(channels), requires_grad=False)
        })
        self.qkv = nn.ParameterDict({
            'weight': nn.Parameter(torch.empty(channels * 3, channels, 1), requires_grad=False),
            'bias': nn.Parameter(torch.empty(channels * 3), requires_grad=False),
        })
        self.proj_out = nn.ParameterDict({
            'weight': nn.Parameter(torch.empty(channels, channels, 1), requires_grad=False),
            'bias': nn.Parameter(torch.empty(channels), requires_grad=False),
        })
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        torda.initialize(
            self.norm.weight,
            self.norm.bias,
            self.qkv.weight,
            self.qkv.bias,
            self.proj_out.weight,
            self.proj_out.bias,
            self.channels,
            self.num_heads,
        )
        return super().load_state_dict(state_dict, strict)

    def forward(self, x):
        return torda.compute(x, self.channels, self.num_heads)
