#pragma once

#include <assert.h>
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
    torch::nn::Conv1dImpl conv(in_channels, out_channels, kernel_size);
    conv.weight = weights;
    conv.bias = bias;

    return conv.forward(x);
}

Tensor attention(Tensor qkv, int n_heads)
{
    auto shape = qkv.sizes();
    int bs = shape[0];
    int width = shape[1];
    int length = shape[2];
    assert(width % (3 * n_heads) == 0);
    int ch = width / (3 * n_heads);

    qkv = qkv.reshape({bs * n_heads, 3 * ch, length});
    Tensor q = qkv.slice(1, 0, ch);
    Tensor k = qkv.slice(1, ch, 2 * ch);
    Tensor v = qkv.slice(1, 2 * ch, 3 * ch);

    float scale = 1. / sqrt(sqrt(ch));
    q *= scale;
    k *= scale;

    q = torch::transpose(q, 1, 2);
    q = torch::matmul(q, k);
    q = torch::softmax(q, -1);

    v = torch::transpose(v, 1, 2);
    Tensor a = torch::matmul(q, v);
    a = torch::transpose(a, 1, 2);

    return a.reshape({bs, -1, length});
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