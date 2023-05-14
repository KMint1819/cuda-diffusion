#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
namespace gten
{
using Tensor = torch::Tensor;
Tensor CUDA_compute(const Tensor x, const Tensor context, const Tensor q_weight, const Tensor k_weight,
                    const Tensor v_weight, const Tensor out_weight, const Tensor out_bias, const int h,
                    const float scale);
} // namespace gten