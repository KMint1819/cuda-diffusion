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
    auto shape = x.sizes();
    int b = shape[0];
    int c = shape[1];
    x.reshape({b, c, -1});

    // Parallelize normalization
    torch::nn::GroupNormImpl norm(32, n_channels);
    norm.weight = norm_weight;
    norm.bias = norm_bias;

    return norm.forward(x);
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