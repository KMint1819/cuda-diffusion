#include "gten_cuda.cuh"
#include <memory>

// input.shape: (m, k)
// weight.shape: (nc, k)
// bias.shape: (nc)
// out.shape: (m, nc)

#define TILE_SZ_A 128
#define TILE_SZ_B 16
#define TILE_SZ_RATIO (TILE_SZ_A / TILE_SZ_B)

__global__ void mysgemm_linear_kernel(int m, int k, int n, const float *A, const float *B, float *bias, float *C,
                                      bool has_bias)
{
    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use register and shared memory tiling and thread coarsening
     *
     ********************************************************************/
//
#define A(row, col) A[(row)*k + (col)]
#define B(row, col) B[(row) + (col)*k]
#define C(row, col) C[(row)*n + (col)]

    __shared__ float sharedN[TILE_SZ_RATIO][TILE_SZ_B];

    int tx = threadIdx.x;

    int sharedX = tx / TILE_SZ_B;
    int sharedY = tx % TILE_SZ_B;

    int x = blockIdx.x * TILE_SZ_A + tx;
    int y = blockIdx.y * TILE_SZ_B;

    float inputMArr[TILE_SZ_RATIO];
    float outputArr[TILE_SZ_B] = {0};

    for (int i = 0; i < ceil(1.0 * k / TILE_SZ_RATIO); i++)
    {

        int Tile_Start = i * TILE_SZ_RATIO;

        if (Tile_Start + sharedX < k && y + sharedY < n)
        {
            sharedN[sharedX][sharedY] = B(Tile_Start + sharedX, y + sharedY);
        }
        __syncthreads();

        for (int j = 0; j < TILE_SZ_RATIO; j++)
        {
            if (x < m && Tile_Start + j < k)
            {
                inputMArr[j] = A(x, Tile_Start + j);
            }
        }
        __syncthreads();

        for (int j = 0; j < TILE_SZ_RATIO; j++)
        {
            for (int out = 0; out < TILE_SZ_B; out++)
            {
                if (x < m && Tile_Start + j < k && y + out < n)
                {
                    outputArr[out] += inputMArr[j] * sharedN[j][out];
                }
            }
        }
        __syncthreads();
    }

    for (int out = 0; out < TILE_SZ_B; out++)
    {
        if (has_bias)
        {
            outputArr[out] += bias[y + out];
        }

        if (x < m && y + out < n)
        {
            C(x, y + out) = outputArr[out];
        }
    }
}

__global__ void basic_linear_kernel(const float *input, const float *weight, const float *bias, float *out, int m,
                                    int k, int n, bool has_bias)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < n && y < m)
    {
        float sum = 0.;
        for (int k = 0; k < k; k++)
        {
            sum += input[y * k + k] * weight[x * k + k];
        }
        if (has_bias)
            sum += bias[x];
        out[y * n + x] = sum;
    }
}

namespace gten
{
using Tensor = torch::Tensor;

/**
 * @brief
 *
 * @param input (1, m, k)
 * @param weight (n, k)
 * @param bias (n) Set bias to empty if you don't want to use bias
 * @return Tensor (1, m, n)
 */
Tensor mysgemm_linear(Tensor input, Tensor weight, Tensor bias)
{

    int m = input.size(1);
    int k = input.size(2);
    int n = weight.size(0);

    Tensor out = torch::zeros({1, m, n}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    bool has_bias = bias.numel() > 0;

    dim3 dimGrid(ceil(1.0 * m / TILE_SZ_A), ceil(1.0 * n / TILE_SZ_B));
    dim3 dimBlock(TILE_SZ_A);

    mysgemm_linear_kernel<<<dimGrid, dimBlock>>>(m, k, n, input.data_ptr<float>(), weight.data_ptr<float>(),
                                                 bias.data_ptr<float>(), out.data_ptr<float>(), has_bias);
    return out;
}

Tensor basic_linear(Tensor input, Tensor weight, Tensor bias)
{
    int m = input.size(1);
    int k = input.size(2);
    int n = weight.size(0);
    Tensor out = torch::zeros({1, m, n}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    bool has_bias = bias.numel() > 0;

    int block_width = 32;
    dim3 grid(ceil(1.0 * n / block_width), ceil(1.0 * m / block_width));
    dim3 block(block_width, block_width);
    basic_linear_kernel<<<grid, block>>>(input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
                                         out.data_ptr<float>(), m, k, n, has_bias);
    return out;
}

Tensor rearrange(Tensor &tensor, int h)
{
    int b = tensor.size(0);
    int n = tensor.size(1);
    int d = tensor.size(2) / h;
    tensor = tensor.reshape({b, n, h, d});
    tensor = tensor.permute({0, 2, 1, 3});
    tensor = tensor.reshape({b * h, n, d});
    return tensor;
}

Tensor CUDA_compute(const Tensor x, const Tensor context, const Tensor q_weight, const Tensor k_weight,
                    const Tensor v_weight, const Tensor out_weight, const Tensor out_bias, const int h,
                    const float scale)
{
    Tensor q = mysgemm_linear(x, q_weight, torch::empty({0}));
    Tensor k = mysgemm_linear(context, k_weight, torch::empty({0}));
    Tensor v = mysgemm_linear(context, v_weight, torch::empty({0}));
    cudaDeviceSynchronize();

    int b = q.size(0);
    int n = q.size(1);
    int d = q.size(2) / h;
    q = rearrange(q, h);
    k = rearrange(k, h);
    v = rearrange(v, h);

    // TODO: replace einsum with mysgemm
    Tensor sim = torch::einsum("b i d, b j d -> b i j", {q, k}) * scale;
    q.reset();
    k.reset();

    sim = sim.softmax(-1);
    Tensor out = torch::einsum("b i j, b j d -> b i d", {sim, v});

    // out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    out = out.reshape({b, h, n, d});
    out = out.permute({0, 2, 1, 3});
    out = out.reshape({b, n, h * d});

    out = mysgemm_linear(out, out_weight, out_bias);
    cudaDeviceSynchronize();

    return out;
}

} // namespace gten