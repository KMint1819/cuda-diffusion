#include "modules.hpp"
#include <iostream>
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <vector>

namespace torda
{
std::string hello(const std::string &name)
{
    return "Saying hello to " + name + " from C++!";
}

torch::OrderedDict<std::string, torch::Tensor> _state_dict;

// TODO: Load state dict instead of parameters
void initialize(Tensor norm_weight, Tensor norm_bias, Tensor qkv_weight, Tensor qkv_bias, Tensor proj_out_weight,
                Tensor proj_out_bias)
{
    printf("Loading the parameters...\n");
    _state_dict.insert("norm.weight", norm_weight);
    _state_dict.insert("norm.bias", norm_bias);
    _state_dict.insert("qkv.weight", qkv_weight);
    _state_dict.insert("qkv.bias", qkv_bias);
    _state_dict.insert("proj_out.weight", proj_out_weight);
    _state_dict.insert("proj_out.bias", proj_out_bias);
    printf("Done loading parameters!\n");
}

Tensor compute(Tensor x, int n_channels, int n_heads)
{
    x = x.cuda();
    auto original_shape = x.sizes();

    // TODO: pass by reference
    x = torda::preprocess(x);
    Tensor norm = torda::normalize(x, _state_dict["norm.weight"], _state_dict["norm.bias"], n_channels);
    Tensor qkv = torda::qkv(norm, _state_dict["qkv.weight"], _state_dict["qkv.bias"], n_channels, n_channels * 3, 1);
    Tensor h = torda::attention(qkv, n_heads);
    h = torda::proj_out(h, _state_dict["proj_out.weight"], _state_dict["proj_out.bias"], n_channels, n_channels, 1);

    return torda::postprocess(x, h, torch::IntArrayRef(original_shape));
}

} // namespace torda

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("hello", &torda::hello, "Say hello from C++!");
    m.def("initialize", &torda::initialize, "Initialize the model");
    m.def("compute", &torda::compute, "Initialize the model");

    // modules.hpp
    m.def("preprocess", &torda::preprocess, "Preprocess the data");
    m.def("normalize", &torda::normalize, "Normalize the data");
    m.def("qkv", &torda::qkv, "Run qkv forward pass");
    m.def("attention", &torda::attention, "Run attention forward pass");
    m.def("proj_out", &torda::proj_out, "Run proj_out(feed forward) forward pass");
    m.def("postprocess", &torda::postprocess, "Postprocess the data");
}
