#pragma once

#define CATCH_CONFIG_CPP11_TO_STRING
#define CATCH_CONFIG_COLOUR_ANSI
#define CATCH_CONFIG_MAIN

#include "common/catch.hpp"
#include "common/fmt.hpp"
#include "common/utils.hpp"

#include "assert.h"
#include "stdint.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include <algorithm>
#include <chrono>
#include <random>
#include <string>
// #include <cstdlib>

#include <cuda.h>

/*********************************************************************/
/* Random number generator                                           */
/* https://en.wikipedia.org/wiki/Xorshift                            */
/* xorshift32                                                        */
/*********************************************************************/

static uint_fast32_t rng_uint32(uint_fast32_t *rng_state)
{
    uint_fast32_t local = *rng_state;
    local ^= local << 13; // a
    local ^= local >> 17; // b
    local ^= local << 5;  // c
    *rng_state = local;
    return local;
}

static uint_fast32_t *rng_new_state(uint_fast32_t seed)
{
    uint64_t *rng_state = new uint64_t;
    *rng_state = seed;
    return rng_state;
}

static uint_fast32_t *rng_new_state()
{
    return rng_new_state(88172645463325252LL);
}

static float rng_float(uint_fast32_t *state)
{
    uint_fast32_t rnd = rng_uint32(state);
    const auto r = static_cast<float>(rnd) / static_cast<float>(UINT_FAST32_MAX);
    if (std::isfinite(r))
    {
        return r;
    }
    return rng_float(state);
}

static int rng_int(uint_fast32_t *state)
{
    uint_fast32_t rnd = rng_uint32(state);
    return static_cast<int>(rnd);
}

static void generate_data(int *x, const int nx, const int ny, const int nz)
{
    const auto rng_state = rng_new_state();

    for (size_t ii = 0; ii < nx * ny * nz; ++ii)
    {
        x[ii] = rng_int(rng_state);
    }

    delete rng_state;
}

static void read_input_data(float *x, const int nx)
{
    // if (__cplusplus == 201103L) puts("C++11");
    // char buffer[100];
    // if (getcwd(buffer, sizeof(buffer)) != NULL) {
    //     printf("Current working directory : %s\n", buffer);
    // }
    // const char* final_command = "ls /../src/data";
    // system(final_command);
    
    FILE* pFile;
    char input_str [32];
    pFile = fopen ("/../src/data/input.txt", "r");
    if (pFile == NULL) perror ("Error opening /../src/data/input.txt");
    else {
        size_t ii = 0;
        while (ii < nx && fgets (input_str , 32 , pFile) != NULL){
            x[ii++] = atof (input_str);
            // puts (input_str);
        }
        fclose (pFile);
        if(ii < nx) perror ("Don't have enough input from /../src/data/input.txt");
    }
}

static bool verify(const float *Anext, const int nx, const int ny, const int nz)
{
    FILE* pFile;
    char out_str [32];
    pFile = fopen ("/../src/data/out.txt", "r");
    if (pFile == NULL) perror ("Error opening /../src/data/out.txt");

    SECTION("the results must match")
    {
        size_t ii = 0;
        while (ii < nx && fgets (out_str , 32 , pFile) != NULL){
            const float expected = atof (out_str);
            INFO("the results did not match at [" << ii << "]");
            REQUIRE(expected == Anext[ii++]);
        }
        if(ii < nx) perror ("Don't have enough input from /../src/data/out.txt");
    }
    fclose (pFile);
    return true;

}

static std::chrono::time_point<std::chrono::high_resolution_clock> now()
{
    return std::chrono::high_resolution_clock::now();
}

#define CUDA_RUNTIME(stmt) checkCuda(stmt, __FILE__, __LINE__);
void checkCuda(cudaError_t result, const char *file, const int line)
{
    if (result != cudaSuccess)
    {
        LOG(critical,
            std::string(fmt::format("{}@{}: CUDA Runtime Error: {}\n", file, line, cudaGetErrorString(result))));
        exit(-1);
    }
}