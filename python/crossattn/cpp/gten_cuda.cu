#include "gten_cuda.cuh"

// input.shape: (nr, nk)
// weight.shape: (nk, nc)
// bias.shape: (nc)
// out.shape: (nr, nc)
__global__ void basic_linear_kernel(const float *input, const float *weight, const float *bias, float *out, int nr,
                                    int nk, int nc, bool has_bias)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    printf("i: %d, j: %d, nr: %d, nc: %d, nk: %d\n", i, j, nr, nc, nk);
    if (i < nr && j < nc)
    {
        float sum = 0.;
        for (int k = 0; k < nk; k++)
        {
            sum += input[i * nk + k] * weight[k * nc + j];
        }
        if (has_bias)
            sum += bias[j];
        if (i * nc + j < nr * nc)
        {
            out[i * nc + j] = sum;
            printf("out[%d, %d] = %f\n", i, j, out[i * nc + j]);
        }
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

    int blockWidth = 32;
    dim3 grid(ceil(1.0 * nr / blockWidth), ceil(1.0 * nc / blockWidth));
    dim3 block(32, 32);
    bool has_bias = bias.numel() > 0;

    Tensor out = torch::zeros({1, nr, nc}, torch::kFloat32);
    basic_linear_kernel<<<grid, block>>>(input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
                                         out.data_ptr<float>(), nr, nk, nc, has_bias);
    cudaDeviceSynchronize();
    return out;
}
} // namespace gten