import torch
from torch import nn

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_dict = nn.ParameterDict({
            'weights': nn.Parameter(torch.ones(3)),
            'bias': nn.Parameter(torch.ones(3)),
        })

mod = MyModule()
print(mod.state_dict())