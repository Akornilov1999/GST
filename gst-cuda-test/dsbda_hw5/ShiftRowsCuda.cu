#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char* file, int line, bool abort=true){
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

__global__ void kernel(int* matrix, int* res_matrix, int count_of_rows, int count_of_columns){
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int column = blockDim.y * blockIdx.y + threadIdx.y;
    if (row < count_of_rows && column < count_of_columns)
        res_matrix[row * count_of_columns + column] = matrix[row * count_of_columns + (column + row) % count_of_columns];
} 

extern "C"  void shiftRows(int* matrix, int* res_matrix, int count_of_rows, int count_of_columns){
    int* cuda_matrix;
    gpuErrchk( cudaMalloc((void**)&cuda_matrix, count_of_rows * count_of_columns * sizeof(int)) );
    int* cuda_res_matrix;
    gpuErrchk( cudaMalloc((void**)&cuda_res_matrix, count_of_rows * count_of_columns * sizeof(int)) );
    gpuErrchk( cudaMemcpy(cuda_matrix, matrix, count_of_rows * count_of_columns * sizeof(int), cudaMemcpyHostToDevice) );
    dim3 threadsPerBlock(count_of_rows, count_of_columns);
    dim3 numBlocks(count_of_rows / threadsPerBlock.x, count_of_columns / threadsPerBlock.y);
    kernel<<<numBlocks, threadsPerBlock>>>(cuda_matrix, cuda_res_matrix, count_of_rows, count_of_columns);
    gpuErrchk( cudaMemcpy(res_matrix, cuda_res_matrix, count_of_rows * count_of_columns * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk( cudaFree(cuda_matrix) );
    gpuErrchk( cudaFree(cuda_res_matrix) );
}
