#include "modules.hpp"
#include <iostream>
#include <torch/extension.h>
#include <vector>

namespace torda
{
std::string hello(const std::string &name)
{
    return "Saying hello to " + name + " from C++!";
}

torch::OrderedDict<std::string, torch::Tensor> _state_dict;
void initialize(Tensor norm_weight, Tensor norm_bias, Tensor qkv_weight, Tensor qkv_bias, Tensor proj_out_weight,
                Tensor proj_out_bias)
{
    _state_dict.insert("norm.weight", norm_weight);
    _state_dict.insert("norm.bias", norm_bias);
    _state_dict.insert("qkv.weight", qkv_weight);
    _state_dict.insert("qkv.bias", qkv_bias);
    _state_dict.insert("proj_out.weight", proj_out_weight);
    _state_dict.insert("proj_out.bias", proj_out_bias);
    auto shape = _state_dict["norm.weight"];
    printf("Shape of norm.weight: \n");
    for (auto it = shape.sizes().begin(); it != shape.sizes().end(); it++)
    {
        printf("%d ", *it);
    }
}

Tensor compute(Tensor input, Tensor norm_weight, Tensor norm_bias, Tensor qkv_weight, Tensor qkv_bias,
               Tensor proj_out_weight, Tensor proj_out_bias, int n_channels, int n_heads)
{
    return {};
}

} // namespace torda

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("hello", &torda::hello, "Say hello from C++!");
    m.def("initialize", &torda::initialize, "Initialize the model");

    // modules.hpp
    m.def("preprocess", &torda::preprocess, "Preprocess the data");
    m.def("normalize", &torda::normalize, "Normalize the data");
    m.def("qkv", &torda::qkv, "Run qkv forward pass");
    m.def("attention", &torda::attention, "Run attention forward pass");
    m.def("proj_out", &torda::proj_out, "Run proj_out(feed forward) forward pass");
    m.def("postprocess", &torda::postprocess, "Postprocess the data");
}
