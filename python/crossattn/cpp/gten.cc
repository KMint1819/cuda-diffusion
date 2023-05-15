#include "gten.hpp"
#include "gten_cuda.cuh"
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

    _layer_to_q = std::make_unique<torch::nn::LinearImpl>(to_q_option);
    _layer_to_k = std::make_unique<torch::nn::LinearImpl>(to_k_option);
    _layer_to_v = std::make_unique<torch::nn::LinearImpl>(to_v_option);
    _layer_to_out = std::make_unique<torch::nn::LinearImpl>(inner_dim, query_dim);

    _layer_to_q->weight.set_requires_grad(false);
    _layer_to_k->weight.set_requires_grad(false);
    _layer_to_v->weight.set_requires_grad(false);
    _layer_to_out->weight.set_requires_grad(false);
    _layer_to_out->bias.set_requires_grad(false);

    _layer_to_q->eval();
    _layer_to_k->eval();
    _layer_to_v->eval();
    _layer_to_out->eval();
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
void CrossAttention::initialize(Tensor to_q_weight, Tensor to_k_weight, Tensor to_v_weight, Tensor to_out_weight,
                                Tensor to_out_bias)
{
    _layer_to_q->weight = std::move(to_q_weight);
    _layer_to_k->weight = std::move(to_k_weight);
    _layer_to_v->weight = std::move(to_v_weight);
    _layer_to_out->weight = std::move(to_out_weight);
    _layer_to_out->bias = std::move(to_out_bias);
}

void CrossAttention::to(torch::Device device)
{
    _device = device;
    _layer_to_q->to(device);
    _layer_to_k->to(device);
    _layer_to_v->to(device);
    _layer_to_out->to(device);
}

// CAUTION: This function modifies the input tensor
Tensor CrossAttention::rearrange(Tensor &tensor, int h) const
{
    int b = tensor.size(0);
    int n = tensor.size(1);
    int d = tensor.size(2) / h;
    tensor = tensor.reshape({b, n, h, d});
    tensor = tensor.permute({0, 2, 1, 3});
    tensor = tensor.reshape({b * h, n, d});
    return tensor;
}

// Tensor CrossAttention::compute(const Tensor &x, const Tensor &context)
// {
//     if (_device == torch::kCPU)
//         _device = torch::kCUDA;
//     to(_device);

//     return CUDA_compute(x, context, _layer_to_q->weight, _layer_to_k->weight, _layer_to_v->weight,
//                         _layer_to_out->weight, _layer_to_out->bias, _heads, _scale);
// }

Tensor CrossAttention::compute(const Tensor &x, const Tensor &context)
{
    if (_device == torch::kCPU)
        _device = torch::kCUDA;
    to(_device);
    auto start_time = std::chrono::high_resolution_clock::now();

    Tensor q = _layer_to_q->forward(x);
    Tensor k = _layer_to_k->forward(context);
    Tensor v = _layer_to_v->forward(context);

    const int h = _heads;
    const int b = q.size(0);
    const int n = q.size(1);
    const int d = q.size(2) / h;
    rearrange(q, h);
    rearrange(k, h);
    rearrange(v, h);

    // TODO: replace einsum with mysgemm. also put the scale as a parameter for the kernel
    Tensor sim = torch::einsum("b i d, b j d -> b i j", {q, k}) * _scale;
    q.reset();
    k.reset();

    sim = sim.softmax(-1);
    Tensor out = torch::einsum("b i j, b j d -> b i d", {sim, v});

    // out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    out = out.reshape({b, h, n, d});
    out = out.permute({0, 2, 1, 3});
    out = out.reshape({b, n, h * d});

    out = _layer_to_out->forward(out);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    printf("Torch C++ compute: %.6f ms\n", duration.count() / 1000.0);
    return out;
}
std::string hello(const std::string &name)
{
    return "Saying hello to " + name + " from C++!";
}
} // namespace gten

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("hello", &gten::hello, "Say hello from C++!");
    py::class_<gten::CrossAttention>(m, "GtenCrossAttention")
        .def(py::init<int, int, int, int, double>(), py::arg("query_dim"), py::arg("context_dim"), py::arg("heads"),
             py::arg("dim_head"), py::arg("dropout"))
        .def("initialize", &gten::CrossAttention::initialize, "Initialize the class with parameters")
        .def("compute", &gten::CrossAttention::compute, "Initialize the model")
        .def("to", &gten::CrossAttention::to, "Move the model to device");
}
