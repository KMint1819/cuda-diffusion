#include "gten_cuda.cuh"

// input.shape: (nr, nk)
// weight.shape: (nk, nc)
// bias.shape: (nc)
// out.shape: (nr, nc)
__global__ void basic_linear_kernel(const float *input, const float *weight, const float *bias, float *out, int nr,
                                    int nk, int nc, bool has_bias)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < nr && c < nc)
    {
        float sum = 0.;
        for (int k = 0; k < nk; k++)
        {
            sum += input[r * nk + k] * weight[k * nc + c];
        }
        if (has_bias)
        {
            sum += bias[c];
        }
        out[r * nc + c] = sum;
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
 * @param weight (nk, nc)
 * @param bias (nc) Set bias to empty if you don't want to use bias
 * @return Tensor (1, nr, nc)
 */
Tensor basic_linear(Tensor input, Tensor weight, Tensor bias)
{
    int nr = input.size(1);
    int nk = input.size(2);
    int nc = weight.size(1);
    printf("input shape: ");
    std::cout << input.sizes() << std::endl;
    printf("weight shape: ");
    std::cout << weight.sizes() << std::endl;
    printf("bias shape: ");
    std::cout << bias.sizes() << std::endl;
    printf("nr: %d, nk: %d, nc: %d\n", nr, nk, nc);

    int block_width = 32;
    dim3 grid(ceil(1.0 * nc / block_width), ceil(1.0 * nr / block_width));
    dim3 block(block_width, block_width);
    bool has_bias = bias.numel() > 0;

    Tensor out = torch::ones({1, nr, nc}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    basic_linear_kernel<<<grid, block>>>(input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
                                         out.data_ptr<float>(), nr, nk, nc, has_bias);
    cudaDeviceSynchronize();
    return out;
}
} // namespace gten