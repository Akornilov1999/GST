#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <time.h>
#define MEGA 1024 * 1024 

int main(int argc, char *argv[]){
    float start_time = (float)clock();
    int size_of_data = 0;
    FILE* fp = fopen("input", "rb");
    int count_of_matrices;
    fread(&count_of_matrices, sizeof(int), 1, fp);
    int*** matrices = (int***) malloc(count_of_matrices * sizeof(int**));
    int** ranks_of_matrices = (int**) malloc(count_of_matrices * sizeof(int*));
    for (int index_of_matrix = 0; index_of_matrix < count_of_matrices; index_of_matrix++){
        ranks_of_matrices[index_of_matrix] = (int*) malloc(2 * sizeof(int));
        int count_of_rows;
        fread(&count_of_rows, sizeof(int), 1, fp);
        ranks_of_matrices[index_of_matrix][0] = count_of_rows;
        int count_of_columns;
        fread(&count_of_columns, sizeof(int), 1, fp);
        ranks_of_matrices[index_of_matrix][1] = count_of_columns;
        matrices[index_of_matrix] = (int**) malloc(count_of_rows * sizeof(int*));
        for (int row = 0; row < count_of_rows; row++){
            matrices[index_of_matrix][row] = (int*) malloc(count_of_columns * sizeof(int));
            for (int column = 0; column < count_of_columns; column++)
                fread(&matrices[index_of_matrix][row][column], sizeof(int), 1, fp);
        }
        size_of_data += count_of_rows * count_of_columns * sizeof(int);
    }
    fclose(fp);
    // float start_time = clock();
    for (int index_of_matrix = 0; index_of_matrix < count_of_matrices; index_of_matrix++)
        for (int row = 0; row < ranks_of_matrices[index_of_matrix][0]; row++){
            int* vector = (int*) malloc(ranks_of_matrices[index_of_matrix][1] * sizeof(int));
            for (int column = 0; column < ranks_of_matrices[index_of_matrix][1]; column++)
                vector[column] = matrices[index_of_matrix][row][column];
            for (int column = 0; column < ranks_of_matrices[index_of_matrix][1]; column++)
                matrices[index_of_matrix][row][column] = vector[(column + row) % ranks_of_matrices[index_of_matrix][1]];
            free(vector);
        }
    // float end_time = ((float)clock()) - start_time;
    fp = fopen("output.txt", "w");
    fprintf(fp, "%d\n", count_of_matrices);
    for (int index_of_matrix = 0; index_of_matrix < count_of_matrices; index_of_matrix++){
        fprintf(fp, "%d %d\n", ranks_of_matrices[index_of_matrix][0], ranks_of_matrices[index_of_matrix][1]);
        for (int row = 0; row < ranks_of_matrices[index_of_matrix][0]; row++){
            for (int column = 0; column < ranks_of_matrices[index_of_matrix][1]; column++)
                fprintf(fp, "%d ", matrices[index_of_matrix][row][column]);
            fputs("\n", fp);
        }
    }
    for (int index_of_matrix = 0; index_of_matrix < count_of_matrices; index_of_matrix++){
        for (int row = 0; row < ranks_of_matrices[index_of_matrix][0]; row++)
            free(matrices[index_of_matrix][row]);
        free(matrices[index_of_matrix]);
    }
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
