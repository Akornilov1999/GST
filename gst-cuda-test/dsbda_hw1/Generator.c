#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <time.h>
#define MEGA 1024 * 1024

void error(char* message){
    printf("%s", message);
    exit(0);
}

int main(int argc, char *argv[]){
    if (argc < 2)
        error("Error! The size of data isn't entered!\n");
    if (atoi(argv[1]) < 1 || atoi(argv[1]) > 650)
        error("Error! The size of data should be from 0 to 650 MB!\n");
    int size_of_data = atoi(argv[1]) * MEGA;
    FILE* fp = fopen("input.txt", "w");
    srand(time(NULL));
    int*** matrices = (int***) malloc(size_of_data / sizeof(int) * sizeof(int**));
    int** ranks_of_matrices = (int**) malloc(size_of_data / sizeof(int) * sizeof(int*));
    int count_of_matrices = 0;
    for(;;){
        int count_of_rows = rand()%9 + 1;
        int count_of_colums = rand()%9 + 1;
        size_of_data -= count_of_rows * count_of_colums * sizeof(int);
        if (size_of_data <= 0)
            break;
        ranks_of_matrices[count_of_matrices] = (int*) malloc(2 * sizeof(int));
        ranks_of_matrices[count_of_matrices][0] = count_of_rows;
        ranks_of_matrices[count_of_matrices][1] = count_of_colums;
        matrices[count_of_matrices] = (int**) malloc(count_of_rows * sizeof(int*));
        for (int row = 0; row < count_of_rows; row++){
            matrices[count_of_matrices][row] = (int*) malloc(count_of_colums * sizeof(int));
            for (int column = 0; column < count_of_colums; column++)
                matrices[count_of_matrices][row][column] = rand()%999 + 1;
        }
        count_of_matrices++;
    }
    fprintf(fp, "%d\n", count_of_matrices);
    for (int index_of_matrix = 0; index_of_matrix < count_of_matrices; index_of_matrix++){
        fprintf(fp, "%d %d\n", ranks_of_matrices[index_of_matrix][0], ranks_of_matrices[index_of_matrix][1]);
        for (int row = 0; row < ranks_of_matrices[index_of_matrix][0]; row++){
            for (int column = 0; column < ranks_of_matrices[index_of_matrix][1]; column++)
                fprintf(fp, "%d ", matrices[index_of_matrix][row][column]);
            fputs("\n", fp);
        }
    }
    fclose(fp);
    fp = fopen("input", "wb");
    fwrite(&count_of_matrices, sizeof(int), 1, fp);
    for (int index_of_matrix = 0; index_of_matrix < count_of_matrices; index_of_matrix++){
        fwrite(&ranks_of_matrices[index_of_matrix][0], sizeof(int), 1, fp);
        fwrite(&ranks_of_matrices[index_of_matrix][1], sizeof(int), 1, fp);
        for (int row = 0; row < ranks_of_matrices[index_of_matrix][0]; row++){
            for (int column = 0; column < ranks_of_matrices[index_of_matrix][1]; column++)
                fwrite(&matrices[index_of_matrix][row][column], sizeof(int), 1, fp);
        }
    }
    fclose(fp);
    for (int index_of_matrix = 0; index_of_matrix < count_of_matrices; index_of_matrix++){
        for (int row = 0; row < ranks_of_matrices[index_of_matrix][0]; row++)
            free(matrices[index_of_matrix][row]);
        free(matrices[index_of_matrix]);
    }
    free(matrices);
    for (int index_of_matrix = 0; index_of_matrix < count_of_matrices; index_of_matrix++)
        free(ranks_of_matrices[index_of_matrix]);
    free(ranks_of_matrices);
    printf("Success!\n");
    return 0;
}
