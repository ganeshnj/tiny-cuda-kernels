#pragma once

// Launches a naive row-major GEMM: C[M, N] = A[M, K] x B[K, N].
void launch_naive_gemm(const float* a, const float* b, float* c,
                       int m, int n, int k);
