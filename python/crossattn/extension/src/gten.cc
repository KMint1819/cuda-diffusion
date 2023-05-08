#include "gten.hpp"
#include <iostream>
#include <memory>
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <vector>

namespace gten
{
std::string hello(const std::string &name)
{
    return "Saying hello to " + name + " from C++!";
}

namespace CrossAttention
{
std::unique_ptr<torch::nn::LinearImpl> layer_to_q;
std::unique_ptr<torch::nn::LinearImpl> layer_to_k;
std::unique_ptr<torch::nn::LinearImpl> layer_to_v;
std::unique_ptr<torch::nn::LinearImpl> layer_to_out;
std::unique_ptr<torch::nn::DropoutImpl> layer_dropout;

// TODO: Load state dict instead of parameters
void initialize(Tensor to_q_weight, Tensor to_k_weight, Tensor to_v_weight, Tensor to_out_weight, Tensor to_out_bias,
                int query_dim, int context_dim, int heads, int dim_head, double dropout)
{
    {
        int inner_dim = heads * dim_head;
        torch::nn::LinearOptions to_q_option(query_dim, inner_dim);
        to_q_option.bias(false);
        torch::nn::LinearOptions to_k_option(query_dim, inner_dim);
        to_k_option.bias(false);
        torch::nn::LinearOptions to_v_option(query_dim, inner_dim);
        to_v_option.bias(false);
        torch::nn::LinearOptions to_out_option(inner_dim, query_dim);
        torch::nn::DropoutOptions dropout_option(dropout);

        layer_to_q = std::make_unique<torch::nn::LinearImpl>(to_q_option);
        layer_to_k = std::make_unique<torch::nn::LinearImpl>(to_k_option);
        layer_to_v = std::make_unique<torch::nn::LinearImpl>(to_v_option);
        layer_to_out = std::make_unique<torch::nn::LinearImpl>(to_out_option);
        layer_dropout = std::make_unique<torch::nn::DropoutImpl>(dropout_option);
    }

    {
        layer_to_q->weight = to_q_weight;
        layer_to_k->weight = to_k_weight;
        layer_to_v->weight = to_v_weight;
        layer_to_out->weight = to_out_weight;
        layer_to_out->bias = to_out_bias;
    }
}

Tensor compute(Tensor x, Tensor context)
{
    printf("Computing...\n");
    x = x.cuda();
    context = context.cuda();
    return context;
}

} // namespace CrossAttention
} // namespace gten

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("hello", &gten::hello, "Say hello from C++!");
    m.def("initialize", &gten::CrossAttention::initialize, "Initialize the model");
    m.def("compute", &gten::CrossAttention::compute, "Initialize the model");

    // // modules.hpp
    // m.def("preprocess", &gten::preprocess, "Preprocess the data");
    // m.def("normalize", &gten::normalize, "Normalize the data");
    // m.def("qkv", &gten::qkv, "Run qkv forward pass");
    // m.def("attention", &gten::attention, "Run attention forward pass");
    // m.def("proj_out", &gten::proj_out, "Run proj_out(feed forward) forward pass");
    // m.def("postprocess", &gten::postprocess, "Postprocess the data");
}
