#include "tck/cuda_utils.cuh"
#include "tck/matmul_gemm.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

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
  int m = parse_int_arg(argc > 1 ? argv[1] : nullptr, 512);
  int n = parse_int_arg(argc > 2 ? argv[2] : nullptr, 512);
  int k = parse_int_arg(argc > 3 ? argv[3] : nullptr, 512);
  int iterations = parse_int_arg(argc > 4 ? argv[4] : nullptr, 20);
  int warmup = parse_int_arg(argc > 5 ? argv[5] : nullptr, 5);
  bool csv = argc > 6 && std::string(argv[6]) == "--csv";

  std::cout << "MatmulNaive M=" << m << " N=" << n << " K=" << k
            << " iterations=" << iterations << " warmup=" << warmup << std::endl;

  std::size_t a_elems = static_cast<std::size_t>(m) * static_cast<std::size_t>(k);
  std::size_t b_elems = static_cast<std::size_t>(k) * static_cast<std::size_t>(n);
  std::size_t c_elems = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);

  std::vector<float> h_a(a_elems, 1.0f);
  std::vector<float> h_b(b_elems, 1.0f);
  std::vector<float> h_c(c_elems, 0.0f);

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  std::size_t a_bytes = a_elems * sizeof(float);
  std::size_t b_bytes = b_elems * sizeof(float);
  std::size_t c_bytes = c_elems * sizeof(float);

  CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
  CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
  CUDA_CHECK(cudaMalloc(&d_c, c_bytes));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), a_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), b_bytes, cudaMemcpyHostToDevice));

  launch_naive_gemm(d_a, d_b, d_c, m, n, k);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, c_bytes, cudaMemcpyDeviceToHost));

  float expected = static_cast<float>(k);
  bool ok = std::all_of(h_c.begin(), h_c.end(), [expected](float v) {
    return std::fabs(v - expected) < 1e-3f;
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
    launch_naive_gemm(d_a, d_b, d_c, m, n, k);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start_evt));
  for (int i = 0; i < iterations; ++i) {
    launch_naive_gemm(d_a, d_b, d_c, m, n, k);
  }
  CUDA_CHECK(cudaEventRecord(stop_evt));
  CUDA_CHECK(cudaEventSynchronize(stop_evt));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_evt, stop_evt));
  double avg_ms = static_cast<double>(elapsed_ms) / static_cast<double>(iterations);

  double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
  double tflops = (flops / (avg_ms / 1000.0)) / 1e12;

  if (csv) {
    std::cout << "m,n,k,iterations,warmup,avg_ms,tflops" << std::endl;
    std::cout << m << "," << n << "," << k << "," << iterations << "," << warmup
              << "," << avg_ms << "," << tflops << std::endl;
  } else {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Correctness: PASS" << std::endl;
    std::cout << "Average kernel time: " << avg_ms << " ms" << std::endl;
    std::cout << "Throughput: " << tflops << " TFLOP/s" << std::endl;
  }

  CUDA_CHECK(cudaEventDestroy(start_evt));
  CUDA_CHECK(cudaEventDestroy(stop_evt));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  return EXIT_SUCCESS;
}
