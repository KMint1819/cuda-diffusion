#pragma once

#include <assert.h>
#include <iostream>
#include <torch/extension.h>
#include <vector>

namespace gten
{
using torch::Tensor;
class CrossAttention
{
  public:
    CrossAttention(int query_dim, int context_dim, int heads, int dim_head, double dropout);

    // TODO: Load state dict instead of parameters
    void loadData(Tensor to_q_weight, Tensor to_k_weight, Tensor to_v_weight, Tensor to_out_weight, Tensor to_out_bias);

    Tensor rearrange(Tensor tensor, int h) const;
    Tensor compute(Tensor x, Tensor context);
    void to(torch::Device device);

  private:
    std::unique_ptr<torch::nn::LinearImpl> _layer_to_q;
    std::unique_ptr<torch::nn::LinearImpl> _layer_to_k;
    std::unique_ptr<torch::nn::LinearImpl> _layer_to_v;
    std::unique_ptr<torch::nn::LinearImpl> _layer_to_out;
    std::unique_ptr<torch::nn::DropoutImpl> _layer_dropout;
    const int _heads;
    const double _scale;
    torch::Device _device = torch::kCPU;
};
} // namespace gten