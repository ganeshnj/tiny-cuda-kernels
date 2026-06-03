#include "tck/cuda_utils.cuh"
#include "tck/vector_add.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::size_t parse_size_arg(const char* arg, std::size_t fallback) {
  if (arg == nullptr) {
    return fallback;
  }
  try {
    return static_cast<std::size_t>(std::stoull(arg));
  } catch (...) {
    return fallback;
  }
}

int parse_int_arg(const char* arg, int fallback) {
  if (arg == nullptr) {
    return fallback;
  }
  try {
    int value = std::stoi(arg);
    return value > 0 ? value : fallback;
  } catch (...) {
    return fallback;
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::size_t n = parse_size_arg(argc > 1 ? argv[1] : nullptr, 1u << 24);
  int iterations = parse_int_arg(argc > 2 ? argv[2] : nullptr, 100);
  int warmup = parse_int_arg(argc > 3 ? argv[3] : nullptr, 10);
  bool csv = argc > 4 && std::string(argv[4]) == "--csv";

  std::cout << "VectorAdd N=" << n << " iterations=" << iterations
            << " warmup=" << warmup << std::endl;

  std::vector<float> h_a(n, 1.25f);
  std::vector<float> h_b(n, 2.50f);
  std::vector<float> h_c(n, 0.0f);

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  std::size_t bytes = n * sizeof(float);

  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  launch_vector_add(d_a, d_b, d_c, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

  float expected = h_a[0] + h_b[0];
  bool ok = std::all_of(h_c.begin(), h_c.end(), [expected](float v) {
    return std::fabs(v - expected) < 1e-6f;
  });

  if (!ok) {
    std::cerr << "Correctness check failed." << std::endl;
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return EXIT_FAILURE;
  }

  cudaEvent_t start_evt{}, stop_evt{};
  CUDA_CHECK(cudaEventCreate(&start_evt));
  CUDA_CHECK(cudaEventCreate(&stop_evt));

  for (int i = 0; i < warmup; ++i) {
    launch_vector_add(d_a, d_b, d_c, n);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start_evt));
  for (int i = 0; i < iterations; ++i) {
    launch_vector_add(d_a, d_b, d_c, n);
  }
  CUDA_CHECK(cudaEventRecord(stop_evt));
  CUDA_CHECK(cudaEventSynchronize(stop_evt));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_evt, stop_evt));
  double avg_ms = static_cast<double>(elapsed_ms) / static_cast<double>(iterations);

  double bytes_per_kernel = static_cast<double>(bytes) * 3.0;
  double gb_per_s = (bytes_per_kernel / (avg_ms / 1000.0)) / 1e9;

  if (csv) {
    std::cout << "n,iterations,warmup,avg_ms,kernel_gbps" << std::endl;
    std::cout << n << "," << iterations << "," << warmup << "," << avg_ms << ","
              << gb_per_s << std::endl;
  } else {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Correctness: PASS" << std::endl;
    std::cout << "Average kernel time: " << avg_ms << " ms" << std::endl;
    std::cout << "Kernel throughput (GB/s): " << gb_per_s << std::endl;
  }

  CUDA_CHECK(cudaEventDestroy(start_evt));
  CUDA_CHECK(cudaEventDestroy(stop_evt));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  return EXIT_SUCCESS;
}
