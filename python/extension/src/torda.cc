#include "modules.hpp"
#include <iostream>
#include <torch/extension.h>
#include <vector>

namespace torda
{

Tensor compute(Tensor input, Tensor norm_weight, Tensor norm_bias, Tensor qkv_weight, Tensor qkv_bias,
               Tensor proj_out_weight, Tensor proj_out_bias, int n_channels, int n_heads)
{
    return {};
}
} // namespace torda

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("preprocess", &torda::preprocess, "Preprocess the data");
    m.def("normalize", &torda::normalize, "Normalize the data");
    m.def("qkv", &torda::qkv, "Run qkv forward pass");
    m.def("attention", &torda::attention, "Run attention forward pass");
    m.def("proj_out", &torda::proj_out, "Run proj_out(feed forward) forward pass");
    m.def("postprocess", &torda::postprocess, "Postprocess the data");
}
