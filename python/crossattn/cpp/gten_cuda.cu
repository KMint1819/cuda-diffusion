#include "gten_cuda.cuh"

// input.shape: (nr, nk)
// weight.shape: (nc, nk)
// bias.shape: (nc)
// out.shape: (nr, nc)
__global__ void basic_linear_kernel(const float *input, const float *weight, const float *bias, float *out, int nr,
                                    int nk, int nc, bool has_bias) {
    
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (x < nc && y < nr) {
        float sum = 0.;
        for (int k = 0; k < nk; k++) {
            sum += input[y * nk + k] * weight[x * nk + k];
        }
        if (has_bias)
            sum += bias[x];
        out[y * nc + x] = sum;
    }
}

namespace gten
{
using Tensor = torch::Tensor;

// TODO: use std::optional
// input shape: (1, nr, nk)
// weight shape: (nk, nc)
// bias shape: nc
/**
 * @brief
 *
 * @param input (1, nr, nk)
 * @param weight (nc, nk)
 * @param bias (nc) Set bias to empty if you don't want to use bias
 * @return Tensor (1, nr, nc)
 */
Tensor basic_linear(Tensor input, Tensor weight, Tensor bias) {
    int nr = input.size(1);
    int nk = input.size(2);
    int nc = weight.size(0);
    // printf("input shape: ");
    // std::cout << input.sizes() << std::endl;
    // printf("weight shape: ");
    // std::cout << weight.sizes() << std::endl;
    // printf("bias shape: ");
    // std::cout << bias.sizes() << std::endl;
    // printf("nr: %d, nk: %d, nc: %d\n", nr, nk, nc);

    int block_width = 32;
    dim3 grid(ceil(1.0 * nc / block_width), ceil(1.0 * nr / block_width));
    dim3 block(block_width, block_width);
    bool has_bias = bias.numel() > 0;
    // printf("has_bias: %d\n\n\n", has_bias);

    Tensor out = torch::zeros({1, nr, nc}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    basic_linear_kernel<<<grid, block>>>(input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
                                         out.data_ptr<float>(), nr, nk, nc, has_bias);
    cudaDeviceSynchronize();
    return out;
}
} // namespace gten