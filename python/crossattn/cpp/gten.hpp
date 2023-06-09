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
    void initialize(Tensor to_q_weight, Tensor to_k_weight, Tensor to_v_weight, Tensor to_out_weight,
                    Tensor to_out_bias);
    Tensor compute(const Tensor &x, const Tensor &context);
    Tensor rearrange(Tensor &tensor, int h) const;
    void to(torch::Device device);

  private:
    std::unique_ptr<torch::nn::LinearImpl> _layer_to_q;
    std::unique_ptr<torch::nn::LinearImpl> _layer_to_k;
    std::unique_ptr<torch::nn::LinearImpl> _layer_to_v;
    std::unique_ptr<torch::nn::LinearImpl> _layer_to_out;
    const int _heads;
    const double _scale;
    torch::Device _device = torch::kCPU;
};
std::string hello(const std::string &name);
} // namespace gten