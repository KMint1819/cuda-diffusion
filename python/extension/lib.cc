#include <iostream>
#include <torch/extension.h>
#include <vector>

torch::Tensor d_sigmoid(torch::Tensor z)
{
    auto s = torch::sigmoid(z);
    return (1 - s) * s;
}

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

int my_add()
{
    return 10 + 20;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &lltm_forward, "LLTM forward");
    m.def("add", &my_add, "My Add!");
}