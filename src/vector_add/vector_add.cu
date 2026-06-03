#include "tck/vector_add.cuh"

#include "tck/cuda_utils.cuh"

namespace {

__global__ void vector_add(const float* a, const float* b, float* c,
                           std::size_t n) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

}  // namespace

void launch_vector_add(const float* a, const float* b, float* c, std::size_t n) {
  constexpr int threads_per_block = 256;
  int blocks = static_cast<int>((n + threads_per_block - 1) / threads_per_block);
  vector_add<<<blocks, threads_per_block>>>(a, b, c, n);
  CUDA_CHECK(cudaGetLastError());
}
