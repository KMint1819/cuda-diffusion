#pragma once

#include <iostream>
#include <torch/extension.h>
#include <vector>

namespace torda
{
using torch::Tensor;
Tensor preprocess(Tensor x, Tensor norm_weight, Tensor norm_bias, int n_channels)
{
    // do reshape and normalization
    return torch::zeros({4, 6});
}

Tensor qkv(Tensor x, Tensor weights, Tensor bias, int in_channels, int out_channels, int kernel_size)
{
    return torch::zeros({4, 6});
}

Tensor attention(Tensor x, int n_heads)
{
    return torch::zeros({4, 6});
}

Tensor proj_out(Tensor x, Tensor weights, Tensor bias, int in_channels, int out_channels, int kernel_size)
{
    return torch::zeros({4, 6});
}

Tensor postprocess(Tensor x, const std::vector<int> &spatial)
{
    return torch::zeros({4, 6});
}

} // namespace torda