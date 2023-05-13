#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
namespace gten
{
using Tensor = torch::Tensor;
Tensor basic_linear(Tensor input, Tensor weight, Tensor bias);
}