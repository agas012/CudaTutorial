#include <iostream>
#include <vector>
#include <algorithm> /*generate*/
#include <chrono>

#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, free, rand */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 10

extern "C" cudaError_t addWithCuda(std::vector<float>&a, std::vector<float>&b, std::vector<float>&c, unsigned int size);

void vector_add_cpu(std::vector<float>& a, std::vector<float>& b, std::vector<float>& out) {
    for (int i = 0; i < N; i++) {
        out[i] = a[i] + b[i];
    }
}

int main() {
    std::vector<float> a_vec(N), b_vec(N), c_vec_cpu(N), c_vec_gpu(N);
    // Generate Random values
    // []() Lambda expressions https://docs.microsoft.com/en-us/cpp/cpp/lambda-expressions-in-cpp?view=msvc-170
    // []() -> captureless lambda as function pointer must have a return
    auto f = []() -> float { return static_cast <float> (rand()) / static_cast <float> (RAND_MAX); };
    // the lambda expression will be used latter to fill vectors, you can see the return as:
    // float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    // another example for integers:
    // auto i = []() -> int { return rand() % 10000; };
    // Fill up the vector
    std::generate(a_vec.begin(), a_vec.end(), f);
    std::generate(b_vec.begin(), b_vec.end(), f);

    auto t1 = std::chrono::high_resolution_clock::now();
    vector_add_cpu(a_vec, b_vec, c_vec_cpu);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout <<"CPU time "  << ms_double.count() << "ms\n";

    // Add vectors in parallel.
    t1 = std::chrono::high_resolution_clock::now();
    cudaError_t cudaStatus = addWithCuda(a_vec, b_vec, c_vec_gpu, N);
    t2 = std::chrono::high_resolution_clock::now();
    ms_double = t2 - t1;
    std::cout << "GPU time " << ms_double.count() << "ms\n";
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
}