#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

static double a = 1.0;

void PrintMatrix(double *matrix, int N, char name[]);
void FillMatrix(double *matrix, int size);
void Multiply(double *matrixA, double *matrixB, double *matrixC, int sizeX, int sizeY);

int main(int argc, char *argv[])
{
    int rank;        //
    int proccess;    //
    int size;        //
    int i;           //
    int waste;       //
    int n;           //
    int processSize; //
    double *matrixA; //
    double *matrixB; //
    double *matrixC; //

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proccess);

    size = 5;
    waste = size % (proccess - 1);
    n = size / (proccess - 1);
    matrixB = (double *)malloc(sizeof(double) * size * size);

    if (rank == 0)
    {
        matrixA = (double *)malloc(sizeof(double) * size * size);
        matrixC = (double *)malloc(sizeof(double) * size * size);
        FillMatrix(matrixA, size);
        FillMatrix(matrixB, size);
    }

    MPI_Bcast(matrixB, size * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {

        processSize = n;

        for (i = 1; i <= (proccess - 1); i++)
        {
            if (i == proccess - 1)
                processSize = waste + n;
            MPI_Send(matrixA + ((i - 1) * size * n), processSize * size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }

        processSize = n;

        for (i = 1; i <= (proccess - 1); i++)
        {
            if (i == proccess - 1)
                processSize = waste + n;
            MPI_Recv(matrixC + ((i - 1) * size * n), processSize * size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        PrintMatrix(matrixA, size, "Matrix A");
        PrintMatrix(matrixB, size, "Matrix B");
        PrintMatrix(matrixC, size, "Matrix C (result)");
    }
    else
    {
        if (rank == proccess - 1)
            processSize = waste + n;
        else
            processSize = n;

        matrixA = (double *)malloc(sizeof(double) * processSize * size);
        matrixC = (double *)malloc(sizeof(double) * processSize * size);

        MPI_Recv(matrixA, processSize * size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        Multiply(matrixA, matrixB, matrixC, processSize, size);
        MPI_Send(matrixC, processSize * size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    free(matrixA);
    free(matrixB);
    free(matrixC);

    MPI_Finalize();
    return 0;
}

void PrintMatrix(double *matrix, int N, char name[])
{
    int i;
    int j;

    printf("%s\n", name);

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
            printf("%.2f\t", matrix[(i * N) + j]);
        printf("\n");
    }
}

void FillMatrix(double *matrix, int size)
{
    int i;
    int j;

    for (i = 0; i < size; i++)
        for (j = 0; j < size; j++)
            matrix[(i * size) + j] = a;
}

void Multiply(double *matrixA, double *matrixB, double *matrixC, int sizeX, int sizeY)
{
    int i;
    int j;
    int k;
    double result;

    for (i = 0; i < sizeX; i++)
        for (j = 0; j < sizeY; j++)
        {
            result = 0.0;
            for (k = 0; k < sizeY; k++)
                result += matrixA[(i * sizeY) + k] * matrixB[(k * sizeY) + j];
            matrixC[(i * sizeY) + j] = result;
        }
}
