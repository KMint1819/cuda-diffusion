import torch
from torch import nn
import torda

class AttentionBlockFunctional(torch.autograd.Function):
    @staticmethod
    def forward(x, norm_dict, qkv_dict, proj_out_dict, n_channels, n_heads):
        b, c, *spatial = x.shape
        x = torda.preprocess(x, norm_dict['weight'], norm_dict['bias'], n_channels)
        return x

        # Try to aggregrate these three in the cuda code like the following: 
        # return ans = torda.compute()
        qkv = torda.qkv(x, qkv_dict['weight'], qkv_dict['bias'], n_channels, n_channels * 3, 1) 
        h = torda.attention(qkv, n_heads)
        h = torda.proj_out(h, proj_out_dict['weight'], proj_out_dict['bias'], n_channels, n_channels, 1)
        ans = torda.postprocess(x + h, (b, c, *spatial))
        return ans

class AttentionBlock(nn.Module):
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
            'weights': nn.Parameter(torch.empty(channels, channels * 3, 1)),
            'bias': nn.Parameter(torch.empty(channels * 3)),
        })
        self.qkv = nn.ParameterDict({
            'weights': nn.Parameter(torch.empty(channels, channels * 3, 1)),
            'bias': nn.Parameter(torch.empty(channels * 3)),
        })
        self.proj_out = nn.ParameterDict({
            'weights': nn.Parameter(torch.empty(channels, channels, 1)),
            'bias': nn.Parameter(torch.empty(channels)),
        })

    def forward(self, x):
        return AttentionBlockFunctional.apply(x, self.norm, self.qkv, self.proj_out, self.proj_out_dict, self.channels, self.num_heads)
