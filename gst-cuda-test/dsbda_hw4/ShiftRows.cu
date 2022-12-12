#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>

#define MEGA 1024 * 1024
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char* file, int line, bool abort=true){
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

__global__ void shiftRows(int* matrix, int* res_matrix, int count_of_rows, int count_of_columns){
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int column = blockDim.y * blockIdx.y + threadIdx.y;
    if (row < count_of_rows && column < count_of_columns)
        res_matrix[row * count_of_columns + column] = matrix[row * count_of_columns + (column + row) % count_of_columns];
}

int main(int argc, char *argv[]){
    float start_time = (float)clock();
    int size_of_data = 0;
    FILE* fp = fopen("input", "rb");
    int count_of_matrices;
    fread(&count_of_matrices, sizeof(int), 1, fp);
    int** matrices = (int**) malloc(count_of_matrices * sizeof(int*));
    int** ranks_of_matrices = (int**) malloc(count_of_matrices * sizeof(int*));
    for (int index_of_matrix = 0; index_of_matrix < count_of_matrices; index_of_matrix++){
        ranks_of_matrices[index_of_matrix] = (int*) malloc(2 * sizeof(int));
        int count_of_rows;
        fread(&count_of_rows, sizeof(int), 1, fp);
        ranks_of_matrices[index_of_matrix][0] = count_of_rows;
        int count_of_columns;
        fread(&count_of_columns, sizeof(int), 1, fp);
        ranks_of_matrices[index_of_matrix][1] = count_of_columns;
        matrices[index_of_matrix] = (int*) malloc(count_of_rows * count_of_columns * sizeof(int));
        for (int row = 0; row < count_of_rows; row++)
            for (int column = 0; column < count_of_columns; column++)
                fread(&matrices[index_of_matrix][row * count_of_columns + column], sizeof(int), 1, fp);
        size_of_data += count_of_rows * count_of_columns * sizeof(int);
    }
    fclose(fp);
    // float start_time = clock();
    for (int index_of_matrix = 0; index_of_matrix < count_of_matrices; index_of_matrix++){
        int* buffer_matrix = (int*) malloc(ranks_of_matrices[index_of_matrix][0] * ranks_of_matrices[index_of_matrix][1] * sizeof(int));
        int* cuda_matrix;
        gpuErrchk( cudaMalloc((void**)&cuda_matrix, ranks_of_matrices[index_of_matrix][0] * ranks_of_matrices[index_of_matrix][1] * sizeof(int)) );
        int* cuda_res_matrix;
        gpuErrchk( cudaMalloc((void**)&cuda_res_matrix, ranks_of_matrices[index_of_matrix][0] * ranks_of_matrices[index_of_matrix][1] * sizeof(int)) );
        gpuErrchk( cudaMemcpy(cuda_matrix, matrices[index_of_matrix], ranks_of_matrices[index_of_matrix][0] * ranks_of_matrices[index_of_matrix][1] * sizeof(int), cudaMemcpyHostToDevice) );
        dim3 threadsPerBlock(ranks_of_matrices[index_of_matrix][0], ranks_of_matrices[index_of_matrix][1]);
        dim3 numBlocks(ranks_of_matrices[index_of_matrix][0] / threadsPerBlock.x, ranks_of_matrices[index_of_matrix][1] / threadsPerBlock.y);
shiftRows<<<numBlocks, threadsPerBlock>>>(cuda_matrix, cuda_res_matrix, ranks_of_matrices[index_of_matrix][0], ranks_of_matrices[index_of_matrix][1]);
        gpuErrchk( cudaMemcpy(buffer_matrix, cuda_res_matrix, ranks_of_matrices[index_of_matrix][0] * ranks_of_matrices[index_of_matrix][1] * sizeof(int), cudaMemcpyDeviceToHost));
        for (int row = 0; row < ranks_of_matrices[index_of_matrix][0]; row++)
            for (int column = 0; column < ranks_of_matrices[index_of_matrix][1]; column++)
                matrices[index_of_matrix][row * ranks_of_matrices[index_of_matrix][1] + column] = buffer_matrix[row * ranks_of_matrices[index_of_matrix][1] + column];
        free(buffer_matrix);
        gpuErrchk( cudaFree(cuda_matrix) );
        gpuErrchk( cudaFree(cuda_res_matrix) );
    }
    fp = fopen("output.txt", "w");
    fprintf(fp, "%d\n", count_of_matrices);
    for (int index_of_matrix = 0; index_of_matrix < count_of_matrices; index_of_matrix++){
        fprintf(fp, "%d %d\n", ranks_of_matrices[index_of_matrix][0], ranks_of_matrices[index_of_matrix][1]);
        for (int row = 0; row < ranks_of_matrices[index_of_matrix][0]; row++){
            for (int column = 0; column < ranks_of_matrices[index_of_matrix][1]; column++)
                fprintf(fp, "%d ", matrices[index_of_matrix][row * ranks_of_matrices[index_of_matrix][1] + column]);
            fputs("\n", fp);
        }
    }
    for (int index_of_matrix = 0; index_of_matrix < count_of_matrices; index_of_matrix++)
        free(matrices[index_of_matrix]);
    free(matrices);
    for (int index_of_matrix = 0; index_of_matrix < count_of_matrices; index_of_matrix++)
        free(ranks_of_matrices[index_of_matrix]);
    free(ranks_of_matrices);
    float end_time = ((float)clock()) - start_time;
    fprintf(fp, "Размер обработанных данных: %f (МБ)\n", (float)size_of_data / (MEGA));
    fprintf(fp, "Время выполнения вычислений: %f (с)\n", end_time / 1000000);
    printf("Success!\n");
    return 0;
}
