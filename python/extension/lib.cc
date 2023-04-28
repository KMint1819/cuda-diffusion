#include "modules.hpp"
#include <iostream>
#include <torch/extension.h>
#include <vector>

namespace torda
{

std::vector<at::Tensor> lltm_forward(torch::Tensor input, torch::Tensor weights, torch::Tensor bias,
                                     torch::Tensor old_h, torch::Tensor old_cell)
{
    auto X = torch::cat({old_h, input}, /*dim=*/1);

    auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));
    auto gates = gate_weights.chunk(3, /*dim=*/1);

    auto input_gate = torch::sigmoid(gates[0]);
    auto output_gate = torch::sigmoid(gates[1]);
    auto candidate_cell = torch::elu(gates[2], /*alpha=*/1.0);

    auto new_cell = old_cell + candidate_cell * input_gate;
    auto new_h = torch::tanh(new_cell) * output_gate;

    return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gate_weights};
}

torch::Tensor attention_forward(torch::Tensor input, torch::Tensor weights, torch::Tensor bias, int n_channels,
                                int n_heads, int n_head_channels)
{
    return {};
}
} // namespace torda

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &torda::lltm_forward, "LLTM forward");
    m.def("preprocess", &torda::preprocess, "Preprocess the data");
    m.def("normalize", &torda::normalize, "Normalize the data");
    m.def("qkv", &torda::qkv, "Run qkv forward pass");
    m.def("attention", &torda::attention, "Run attention forward pass");
    m.def("proj_out", &torda::proj_out, "Run proj_out(feed forward) forward pass");
    m.def("postprocess", &torda::postprocess, "Postprocess the data");
}
