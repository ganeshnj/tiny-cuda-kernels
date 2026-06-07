cmake_minimum_required(VERSION 3.18)

project(tiny_cuda_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

add_library(tck_kernels STATIC
  src/matmul/01_naive_gemm.cu
  src/vector_add/vector_add.cu
)

target_include_directories(tck_kernels
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(tck_kernels
  PUBLIC
    CUDA::cudart
)

set_target_properties(tck_kernels PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

add_executable(vector_add
  src/vector_add_main.cu
)

add_executable(matmul_naive
  src/matmul_naive_main.cu
)

target_include_directories(vector_add
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(vector_add
  PRIVATE
    tck_kernels
    CUDA::cudart
)

target_link_libraries(matmul_naive
  PRIVATE
    tck_kernels
    CUDA::cudart
)

set_target_properties(vector_add PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(matmul_naive PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
