#include "tck/matmul_gemm.cuh"

#include "tck/cuda_utils.cuh"

namespace {

__global__ void naive_gemm_kernel(const float* a, const float* b, float* c,
                                  int m, int n, int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= m || col >= n) {
    return;
  }

  // Compute C[row, col] = A[row, :] x B[:, col].
  float acc = 0.0f;
  for (int i = 0; i < k; ++i) {
    // Row-major indexing.
    acc += a[row * k + i] * b[i * n + col];
  }

  // Row-major indexing.
  c[row * n + col] = acc;
}

}  // namespace

void launch_naive_gemm(const float* a, const float* b, float* c,
                       int m, int n, int k) {
  constexpr int tile_x = 16;
  constexpr int tile_y = 16;
  dim3 block(tile_x, tile_y);
  dim3 grid((n + tile_x - 1) / tile_x, (m + tile_y - 1) / tile_y);

  naive_gemm_kernel<<<grid, block>>>(a, b, c, m, n, k);
  CUDA_CHECK(cudaGetLastError());
}
