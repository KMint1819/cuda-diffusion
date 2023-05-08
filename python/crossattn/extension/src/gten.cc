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

CrossAttention::CrossAttention(int query_dim, int context_dim, int heads, int dim_head, double dropout)
    : _heads(heads), _scale(std::pow(dim_head, -0.5))
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

    _layer_to_q = std::make_unique<torch::nn::LinearImpl>(to_q_option);
    _layer_to_k = std::make_unique<torch::nn::LinearImpl>(to_k_option);
    _layer_to_v = std::make_unique<torch::nn::LinearImpl>(to_v_option);
    _layer_to_out = std::make_unique<torch::nn::LinearImpl>(to_out_option);
    _layer_dropout = std::make_unique<torch::nn::DropoutImpl>(dropout_option);
}

// TODO: Load state dict instead of parameters
void CrossAttention::loadData(Tensor to_q_weight, Tensor to_k_weight, Tensor to_v_weight, Tensor to_out_weight,
                              Tensor to_out_bias)
{
    _layer_to_q->weight = to_q_weight;
    _layer_to_k->weight = to_k_weight;
    _layer_to_v->weight = to_v_weight;
    _layer_to_out->weight = to_out_weight;
    _layer_to_out->bias = to_out_bias;
}

Tensor CrossAttention::rearrange(Tensor tensor, int h) const
{
    int b = tensor.size(0);
    int n = tensor.size(1);
    int d = tensor.size(2) / h;
    tensor = tensor.reshape({b, n, h, d});
    tensor = tensor.permute({0, 2, 1, 3});
    tensor = tensor.reshape({b * h, n, d});
    return tensor;
}
Tensor CrossAttention::compute(Tensor x, Tensor context)
{
    printf("Computing...\n");

    int h = _heads;
    Tensor q = _layer_to_q->forward(x);
    Tensor k = _layer_to_k->forward(context);
    Tensor v = _layer_to_v->forward(context);

    int b = q.size(0);
    int n = q.size(1);
    int d = q.size(2) / h;
    q = rearrange(q, h);
    k = rearrange(k, h);
    v = rearrange(v, h);

    Tensor sim = torch::einsum("b i d, b j d -> b i j", {q, k}) * _scale;
    q.reset();
    k.reset();

    sim = sim.softmax(-1);
    Tensor out = torch::einsum("b i j, b j d -> b i d", {sim, v});

    // TODO: solve segfault
    // out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    out = out.reshape({b, h, n, d});
    out = out.permute({0, 2, 1, 3});
    out = out.reshape({b, n, h * d});

    out = _layer_to_out->forward(out);
    return out;
}
} // namespace gten

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("hello", &gten::hello, "Say hello from C++!");
    py::class_<gten::CrossAttention>(m, "GtenCrossAttention")
        .def(py::init<int, int, int, int, double>(), py::arg("query_dim"), py::arg("context_dim"), py::arg("heads"),
             py::arg("dim_head"), py::arg("dropout"))
        // .def("rearrange", &gten::CrossAttention::rearrange, "Rearrange the tensor")
        .def("loadData", &gten::CrossAttention::loadData, "Initialize the model")
        .def("compute", &gten::CrossAttention::compute, "Initialize the model");

    // // modules.hpp
    // m.def("preprocess", &gten::preprocess, "Preprocess the data");
    // m.def("normalize", &gten::normalize, "Normalize the data");
    // m.def("qkv", &gten::qkv, "Run qkv forward pass");
    // m.def("attention", &gten::attention, "Run attention forward pass");
    // m.def("proj_out", &gten::proj_out, "Run proj_out(feed forward) forward pass");
    // m.def("postprocess", &gten::postprocess, "Postprocess the data");
}
