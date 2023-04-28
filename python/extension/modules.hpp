#pragma once

#include <iostream>
#include <torch/extension.h>
#include <vector>

namespace torda
{

using torch::Tensor;
Tensor preprocess(Tensor x)
{
}

Tensor normalize(Tensor x)
{
}

Tensor qkv(Tensor x, Tensor weights, Tensor bias, int n_channels)
{
}

Tensor attention(Tensor x, Tensor weights, Tensor bias, int n_channels, int n_heads, int n_head_channels)
{
}

Tensor proj_out(Tensor x, Tensor weights, Tensor bias, int n_channels)
{
}

Tensor postprocess(Tensor x, const std::vector<int> &spatial)
{
}

} // namespace torda