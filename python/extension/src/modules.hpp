#pragma once

#include <assert.h>
#include <iostream>
#include <torch/extension.h>
#include <vector>

namespace torda
{
using torch::Tensor;
extern std::unique_ptr<torch::nn::GroupNormImpl> norm_layer;
extern std::unique_ptr<torch::nn::Conv1dImpl> qkv_layer;
extern std::unique_ptr<torch::nn::Conv1dImpl> proj_out_layer;
void preprocess(Tensor &x)
{
    // do reshape and normalization
    auto shape = x.sizes();
    int b = shape[0];
    int c = shape[1];

    x.reshape({b, c, -1});
}

Tensor normalize(const Tensor &x)
{
    // Parallelize normalization
    return norm_layer->forward(x);
}

Tensor qkv(const Tensor &x)
{
    return qkv_layer->forward(x);
}

// CAUTION: This function modifies the input tensor
Tensor attention(Tensor &qkv, int n_heads)
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

    // q = torch::einsum("bct,bcs->bts", {q, k});
    q = torch::transpose(q, 1, 2);
    q = torch::matmul(q, k);
    q = torch::softmax(q, -1);

    // v = torch::einsum("bts,bcs->bct", {q, v});
    v = torch::transpose(v, 1, 2);
    v = torch::matmul(q, v);
    v = torch::transpose(v, 1, 2);

    return v.reshape({bs, -1, length});
}

Tensor proj_out(const Tensor &x)
{
    return proj_out_layer->forward(x);
}

Tensor postprocess(const Tensor &x, const Tensor &h, torch::IntArrayRef shape)
{
    return (x + h).reshape(shape);
}

} // namespace torda