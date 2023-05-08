#include "modules.hpp"
#include <iostream>
#include <memory>
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

std::unique_ptr<torch::nn::GroupNormImpl> norm_layer;
std::unique_ptr<torch::nn::Conv1dImpl> qkv_layer;
std::unique_ptr<torch::nn::Conv1dImpl> proj_out_layer;

// TODO: Load state dict instead of parameters
void initialize(Tensor norm_weight, Tensor norm_bias, Tensor qkv_weight, Tensor qkv_bias, Tensor proj_out_weight,
                Tensor proj_out_bias, int n_channels, int n_heads)
{
    norm_layer = std::make_unique<torch::nn::GroupNormImpl>(32, n_channels);
    qkv_layer = std::make_unique<torch::nn::Conv1dImpl>(n_channels, n_channels * 3, 1);
    proj_out_layer = std::make_unique<torch::nn::Conv1dImpl>(n_channels, n_channels, 1);

    norm_layer->weight = norm_weight;
    norm_layer->bias = norm_bias;
    qkv_layer->weight = qkv_weight;
    qkv_layer->bias = qkv_bias;
    proj_out_layer->weight = proj_out_weight;
    proj_out_layer->bias = proj_out_bias;
}

Tensor compute(Tensor x, int n_channels, int n_heads)
{
    printf("Computing...\n");
    x = x.cuda();
    auto original_shape = x.sizes();
    torda::preprocess(x);

    Tensor tensor = torda::normalize(x);
    tensor = torda::qkv(tensor);
    tensor = torda::attention(tensor, n_heads);
    tensor = torda::proj_out(tensor);
    return torda::postprocess(x, tensor, torch::IntArrayRef(original_shape));
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
