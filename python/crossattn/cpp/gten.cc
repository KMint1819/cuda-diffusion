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
    torch::nn::LinearOptions to_k_option(context_dim, inner_dim);
    to_k_option.bias(false);
    torch::nn::LinearOptions to_v_option(context_dim, inner_dim);
    to_v_option.bias(false);
    torch::nn::LinearOptions to_out_0_option(inner_dim, query_dim);
    to_out_0_option.bias(true);

    _layer_to_q = std::make_unique<torch::nn::LinearImpl>(to_q_option);
    _layer_to_k = std::make_unique<torch::nn::LinearImpl>(to_k_option);
    _layer_to_v = std::make_unique<torch::nn::LinearImpl>(to_v_option);
    _layer_to_out_0 = std::make_unique<torch::nn::LinearImpl>(to_out_0_option);
    _layer_to_out_1 = std::make_unique<torch::nn::DropoutImpl>(dropout);

    // print params
    // printf("=========================================\n");
    // printf("Torch version: %d.%d.%d\n", TORCH_VERSION_MAJOR, TORCH_VERSION_MINOR, TORCH_VERSION_PATCH);
    // printf("query_dim: %d\n", query_dim);
    // printf("context_dim: %d\n", context_dim);
    // printf("heads: %d\n", heads);
    // printf("dim_head: %d\n", dim_head);
    // printf("dropout: %f\n", dropout);
    // printf("inner_dim: %d\n", inner_dim);
    // printf("=========================================\n");
}

// TODO: Load state dict instead of parameters
void CrossAttention::loadData(Tensor to_q_weight, Tensor to_k_weight, Tensor to_v_weight, Tensor to_out_0_weight,
                              Tensor to_out_0_bias)
{
    printf("Type of to_q_weight: %s\n", to_q_weight.dtype().name());
    printf("Type of to_k_weight: %s\n", to_k_weight.dtype().name());
    printf("Type of to_v_weight: %s\n", to_v_weight.dtype().name());
    printf("Type of to_out_0_weight: %s\n", to_out_0_weight.dtype().name());
    printf("Type of to_out_0_bias: %s\n", to_out_0_bias.dtype().name());

    // Transfer to gpu
    _layer_to_q->weight = to_q_weight;
    _layer_to_k->weight = to_k_weight;
    _layer_to_v->weight = to_v_weight;
    _layer_to_out_0->weight = to_out_0_weight;
    _layer_to_out_0->bias = to_out_0_bias;
}

void CrossAttention::to(torch::Device device)
{
    _device = device;
    _layer_to_q->to(device);
    _layer_to_k->to(device);
    _layer_to_v->to(device);
    _layer_to_out_0->to(device);
    _layer_to_out_1->to(device);
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

Tensor CrossAttention::compute(Tensor x, Tensor context, Tensor to_q_weight, Tensor to_k_weight, Tensor to_v_weight,
                               Tensor to_out_0_weight, Tensor to_out_0_bias)
{
    loadData(to_q_weight, to_k_weight, to_v_weight, to_out_0_weight, to_out_0_bias);
    printf("Data is loaded!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    _layer_to_q->eval();
    _layer_to_k->eval();
    _layer_to_v->eval();
    _layer_to_out_0->eval();
    _layer_to_out_1->eval();
    torch::NoGradGuard nograd;

    printf("Computing...\n");
    if (_device == torch::kCPU)
        _device = torch::kCUDA;
    to(_device);

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

    // out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    out = out.reshape({b, h, n, d});
    out = out.permute({0, 2, 1, 3});
    out = out.reshape({b, n, h * d});
    // Output above this line is correct

    out = _layer_to_out_0->forward(out);
    out = _layer_to_out_1->forward(out);
    return out;
}
} // namespace gten

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("hello", &gten::hello, "Say hello from C++!");
    py::class_<gten::CrossAttention>(m, "GtenCrossAttention")
        .def(py::init<int, int, int, int, double>(), py::arg("query_dim"), py::arg("context_dim"), py::arg("heads"),
             py::arg("dim_head"), py::arg("dropout"))
        .def("compute", &gten::CrossAttention::compute, "Initialize the model")
        .def("to", &gten::CrossAttention::to, "Move the model to device");
}
