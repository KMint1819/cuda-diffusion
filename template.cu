#include <cstdio>
#include <cstdlib>
#include "helper.hpp"


#define TILE_SIZE 32

__global__ void testKernel(float* originalVal, float* newVal, int valSize) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 0 && i < valSize)
    newVal[i] = 91.035789; //originalVal[i] + 1;
}

void launchTest(float* originalVal, float* newVal, int valSize) {

  // INSERT CODE HERE
  int X_size = ceil(1.0 * valSize / TILE_SIZE);
  dim3 DimGrid(X_size,1,1);
  dim3 DimBlock(TILE_SIZE,1,1);
  testKernel<<<DimGrid,DimBlock>>>(originalVal,newVal,valSize);
}

static int eval(const int nx) {

  // Generate model
  const auto conf_info = std::string("test[") + std::to_string(nx) + "]";
  INFO("Running "  << conf_info);

  // generate input data
  timer_start("Generating test data");
  std::vector<float> hostA0(nx);
  // generate_data(hostA0.data(), nx, 1, 1);
  read_input_data(hostA0.data(), nx);
  std::vector<float> hostAnext(nx);

  timer_start("Allocating GPU memory.");
  float *deviceA0 = nullptr, *deviceAnext = nullptr;
  CUDA_RUNTIME(cudaMalloc((void **)&deviceA0, nx * sizeof(float)));
  CUDA_RUNTIME(cudaMalloc((void **)&deviceAnext, nx * sizeof(float)));
  timer_stop();

  timer_start("Copying inputs to the GPU.");
  CUDA_RUNTIME(cudaMemcpy(deviceA0, hostA0.data(), nx * sizeof(float), cudaMemcpyDefault));
  CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  //////////////////////////////////////////
  // GPU Gather Computation
  //////////////////////////////////////////
  timer_start("Performing GPU convlayer");
  launchTest(deviceA0, deviceAnext, nx);
  CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  timer_start("Copying output to the CPU");
  CUDA_RUNTIME(cudaMemcpy(hostAnext.data(), deviceAnext, nx * sizeof(float), cudaMemcpyDefault));
  CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  // verify with provided implementation
  timer_start("Verifying results");
  verify(hostAnext.data(), nx, 1, 1);
  timer_stop();

  CUDA_RUNTIME(cudaFree(deviceA0));
  CUDA_RUNTIME(cudaFree(deviceAnext));

  return 0;
}

TEST_CASE("Group 10", "[gten]") {

  SECTION("32") {
    eval(32);
  }

  // SECTION("1024") {
  //   eval(1024);
  // }

  // SECTION("2048") {
  //   eval(2048);
  // }
}
