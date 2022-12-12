#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <time.h>
#include "mpi.h"
#define MEGA 1024 * 1024

int main(int argc, char *argv[]){
    float start_time = (float)clock();
    float end_time;
    int rc, rank, number_of_tasks;;
    if ((rc = MPI_Init(&argc, &argv)) != MPI_SUCCESS){
        printf("Error starting MPI programm, Terminating!\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &number_of_tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0){
        int** matrices;
        int** rangs_of_matrices;
        int size_of_data = 0;
        int count_of_matrices;
        FILE* fp;
        fp = fopen("input", "rb");
        fread(&count_of_matrices, sizeof(int), 1, fp);
        matrices = (int**) malloc(count_of_matrices * sizeof(int*));
        rangs_of_matrices = (int**) malloc(2 * sizeof(int*));
        rangs_of_matrices[0] = (int*) malloc(count_of_matrices * sizeof(int));
        rangs_of_matrices[1] = (int*) malloc(count_of_matrices * sizeof(int));
        for (int index_of_matrix = 0; index_of_matrix < count_of_matrices; index_of_matrix++){
            int count_of_rows;
            fread(&count_of_rows, sizeof(int), 1, fp);
            rangs_of_matrices[0][index_of_matrix] = count_of_rows;
            int count_of_columns;
            fread(&count_of_columns, sizeof(int), 1, fp);
            rangs_of_matrices[1][index_of_matrix] = count_of_columns;
            matrices[index_of_matrix] = (int*) malloc(count_of_rows * count_of_columns * sizeof(int));
            for (int row = 0; row < count_of_rows; row++)
                for (int column = 0; column < count_of_columns; column++)
                    fread(&matrices[index_of_matrix][row * count_of_columns + column], sizeof(int), 1, fp);
            size_of_data += count_of_rows * count_of_columns * sizeof(int);
        }
        fclose(fp);
        // float start_time = (float)clock();
        MPI_Status status;
        for (int number_of_rank = 1; number_of_rank < count_of_matrices % (number_of_tasks - 1) + 1; number_of_rank++)
        {
            int message = count_of_matrices / (number_of_tasks - 1) + 1;
            MPI_Send(&message, 1, MPI_INT, number_of_rank, 0, MPI_COMM_WORLD);
        }
        for (int number_of_rank = count_of_matrices % (number_of_tasks - 1) + 1; number_of_rank < number_of_tasks; number_of_rank++)
        {
            int message = count_of_matrices / (number_of_tasks - 1);
            MPI_Send(&message, 1, MPI_INT, number_of_rank, 0, MPI_COMM_WORLD);
        }
        for (int index_of_set = 0; index_of_set < count_of_matrices / (number_of_tasks - 1) * (number_of_tasks - 1); index_of_set += (number_of_tasks - 1))
        {
            for (int number_of_rank = 1; number_of_rank < number_of_tasks; number_of_rank++)
            {
                int* rang_of_matrix = (int*) malloc(2 * sizeof(int));
                rang_of_matrix[0] = rangs_of_matrices[0][index_of_set + number_of_rank - 1];
                rang_of_matrix[1] = rangs_of_matrices[1][index_of_set + number_of_rank - 1];
                MPI_Send(rang_of_matrix, 2, MPI_INT, number_of_rank, 0, MPI_COMM_WORLD);
                MPI_Send(matrices[index_of_set + number_of_rank - 1], rang_of_matrix[0] * rang_of_matrix[1], MPI_INT, number_of_rank, 0, MPI_COMM_WORLD);
                free(rang_of_matrix);
            }
            for (int number_of_rank = 1; number_of_rank < number_of_tasks; number_of_rank++)
            {
                MPI_Status status;
                int* rang_of_matrix = (int*) malloc(2 * sizeof(int));
                rang_of_matrix[0] = rangs_of_matrices[0][index_of_set + number_of_rank - 1];
                rang_of_matrix[1] = rangs_of_matrices[1][index_of_set + number_of_rank - 1];
                MPI_Recv(matrices[index_of_set + number_of_rank - 1], rang_of_matrix[0] * rang_of_matrix[1], MPI_INT, number_of_rank, 0, MPI_COMM_WORLD, &status);
                free(rang_of_matrix);
            }
        }
        for (int number_of_rank = 1; number_of_rank < count_of_matrices % (number_of_tasks - 1) + 1; number_of_rank++)
        {
            int* rang_of_matrix = (int*) malloc(2 * sizeof(int));
            rang_of_matrix[0] = rangs_of_matrices[0][count_of_matrices / (number_of_tasks - 1) * (number_of_tasks - 1) + number_of_rank - 1];
            rang_of_matrix[1] = rangs_of_matrices[1][count_of_matrices / (number_of_tasks - 1) * (number_of_tasks - 1) + number_of_rank - 1];
            MPI_Send(rang_of_matrix, 2, MPI_INT, number_of_rank, 0, MPI_COMM_WORLD);
            MPI_Send(matrices[count_of_matrices / (number_of_tasks - 1) * (number_of_tasks - 1) + number_of_rank - 1], rang_of_matrix[0] * rang_of_matrix[1], MPI_INT, number_of_rank, 0, MPI_COMM_WORLD);
            free(rang_of_matrix);
        }
        for (int number_of_rank = 1; number_of_rank < count_of_matrices % (number_of_tasks - 1) + 1; number_of_rank++)
        {
            MPI_Status status;
            int* rang_of_matrix = (int*) malloc(2 * sizeof(int));
            rang_of_matrix[0] = rangs_of_matrices[0][count_of_matrices / (number_of_tasks - 1) * (number_of_tasks - 1) + number_of_rank - 1];
            rang_of_matrix[1] = rangs_of_matrices[1][count_of_matrices / (number_of_tasks - 1) * (number_of_tasks - 1) + number_of_rank - 1];
            MPI_Recv(matrices[count_of_matrices / (number_of_tasks - 1) * (number_of_tasks - 1) + number_of_rank - 1], rang_of_matrix[0] * rang_of_matrix[1], MPI_INT, number_of_rank, 0, MPI_COMM_WORLD, &status);
            free(rang_of_matrix);
        }
        // float end_time = ((float)clock()) - start_time;
        fp = fopen("output.txt", "w");
        fprintf(fp, "%d\n", count_of_matrices);
        for (int index_of_matrix = 0; index_of_matrix < count_of_matrices; index_of_matrix++){
            fprintf(fp, "%d %d\n", rangs_of_matrices[0][index_of_matrix], rangs_of_matrices[1][index_of_matrix]);
            for (int row = 0; row < rangs_of_matrices[0][index_of_matrix]; row++){
                for (int column = 0; column < rangs_of_matrices[1][index_of_matrix]; column++)
                    fprintf(fp, "%d ", matrices[index_of_matrix][row *rangs_of_matrices[1][index_of_matrix] + column]);
                fputs("\n", fp);
            }
        }
        for (int index_of_matrix = 0; index_of_matrix < count_of_matrices; index_of_matrix++)
            free(matrices[index_of_matrix]);
        free(matrices);
        free(rangs_of_matrices[0]);
        free(rangs_of_matrices[1]);
        free(rangs_of_matrices);
        end_time = ((float)clock()) - start_time;
        fprintf(fp, "Размер обработанных данных: %f (МБ)\n", (float)size_of_data / (MEGA));
        fprintf(fp, "Время выполнения вычислений: %f (с)\n", end_time / 1000000);
        fclose(fp);
        printf("Success!\n");
    }
    else{
        MPI_Status status;
        int count_of_matrices = 0;
        MPI_Recv(&count_of_matrices, sizeof(int), MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        for (int index_of_matrix = 0; index_of_matrix < count_of_matrices; index_of_matrix++){
            int* rank_of_matrix = (int*) malloc(2 * sizeof(int));
            MPI_Recv(rank_of_matrix, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            int* matrix = (int*) malloc(rank_of_matrix[0] * rank_of_matrix[1] * sizeof(int));
            int* buffer_matrix = (int*) malloc(rank_of_matrix[0] * rank_of_matrix[1] * sizeof(int));
            MPI_Recv(buffer_matrix, rank_of_matrix[0] * rank_of_matrix[1], MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            for (int row = 0; row < rank_of_matrix[0]; row++)
            {
                for (int column = 0; column < rank_of_matrix[1]; column++)
                    matrix[row * rank_of_matrix[1] + column] = buffer_matrix[row * rank_of_matrix[1] + (column + row) % rank_of_matrix[1]];
            }
            MPI_Send(matrix, rank_of_matrix[0] * rank_of_matrix[1], MPI_INT, 0, 0, MPI_COMM_WORLD);
            free(buffer_matrix);
            free(matrix);
            free(rank_of_matrix);
        }
    }
    MPI_Finalize();
    return 0;
}
