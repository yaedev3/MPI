/* File: matmult-mpi.c
 *
 * Purpose: 
 * 
 * Input:
 * 
 * Output:
 * 
 * Compile: mpicc -o matmult-mpi-c.o matmult-mpi.c
 * 
 * Run: mpiexec -np 3 ./matmult-mpi-c.o
 * 
 * Algorithm:
 * 
 * Note:
 * 
 * */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"

void PrintMatrix(float *matrix, int row, int column, char name[]);
void FillMatrix(float *matrix, int size);
void Multiply(float *matrixA, float *matrixB, float *matrixC, int sizeX, int sizeY);

int main(int argc, char *argv[])
{
    int rank;                   //
    int proccess;               //
    int size = 5;               //
    int i;                      //
    int waste;                  //
    int n;                      //
    int processSize;            //
    float matrixA[size * size]; //
    float matrixB[size * size]; //
    float matrixC[size * size]; //

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proccess);

    waste = size % (proccess - 1);
    n = size / (proccess - 1);

    if (rank == 0)
    {
        FillMatrix(matrixA, size);
        FillMatrix(matrixB, size);
    }

    MPI_Bcast(matrixA, size * size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(matrixB, size * size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        processSize = n;

        for (i = 1; i <= (proccess - 1); i++)
        {
            if (i == proccess - 1)
                processSize = waste + n;
            MPI_Recv(matrixC + ((i - 1) * size * n), processSize * size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        PrintMatrix(matrixA, size, size, "Matrix A");
        PrintMatrix(matrixB, size, size, "Matrix B");
        PrintMatrix(matrixC, size, size, "Matrix C (result)");
    }
    else
    {
        if (rank == proccess - 1)
            processSize = waste + n;
        else
            processSize = n;

        Multiply(matrixA, matrixB, matrixC, processSize, size);
        MPI_Send(matrixC, processSize * size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }

    free(matrixA);
    free(matrixB);
    free(matrixC);

    MPI_Finalize();
    return 0;
}

void PrintMatrix(float *matrix, int row, int column, char name[])
{
    int i;
    int j;

    printf("%s\n", name);

    for (i = 0; i < row; i++)
    {
        for (j = 0; j < column; j++)
            printf("%.2f\t", matrix[(i * column) + j]);
        printf("\n");
    }
}

void FillMatrix(float *matrix, int size)
{
    int i;
    int j;

    for (i = 0; i < size; i++)
        for (j = 0; j < size; j++)
            matrix[(i * size) + j] = rand() % (11 * (i + 1)) * 1.12;
}

void Multiply(float *matrixA, float *matrixB, float *matrixC, int sizeX, int sizeY)
{
    int i;
    int j;
    int k;
    float result;

    for (i = 0; i < sizeX; i++)
        for (j = 0; j < sizeY; j++)
        {
            result = 0.0;
            for (k = 0; k < sizeY; k++)
                result += matrixA[(i * sizeY) + k] * matrixB[(k * sizeY) + j];
            matrixC[(i * sizeY) + j] = result;
        }
}