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
    if (i < nr && j < nc)
    {
        float sum = 0.;
        for (int k = 0; k < nk; k++)
        {
            sum += input[i * nk + k] * weight[k * nc + j];
        }
        if (has_bias)
            out[i * nc + j] = sum + bias[j];
        printf("out[%d, %d] = %f\n", i, j, out[i * nc + j]);
    }
}

namespace gten
{
using Tensor = torch::Tensor;
Tensor basic_linear(Tensor input, Tensor weight, Tensor bias)
{
    int nr = input.size(0);
    int nk = input.size(1);
    int nc = weight.size(1);
    int blockWidth = 32;
    dim3 grid(ceil(1.0 * nr / blockWidth), ceil(1.0 * nc / blockWidth));
    dim3 block(32, 32);
    bool has_bias = bias.numel() > 0;

    Tensor out = torch::zeros({nr, nc}, torch::kFloat32);
    float *a;
    basic_linear_kernel<<<grid, block>>>(input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
                                         out.data_ptr<float>(), nr, nk, nc, has_bias);
    cudaDeviceSynchronize();
    return out;
}
} // namespace gten