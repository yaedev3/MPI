#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

static double a = 1.0E-10;

void PrintMatrix(double *matrix, int row, int column, char name[]);
void FillMatrix(double *matrix, int N);
void Multiply(double *matrixA, double *matrixB, double *matrixC, int NX, int NY);

int main(int argc, char *argv[])
{
    int rank;                   //
    int proccess;               //
    int N = 3;                  //
    int i;                      //
    int waste;                  //
    int n;                      //
    int processN;               //
    double matrixA[N * N];      //
    double matrixB[N * N];      //
    double matrixC[N * N];      //

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proccess);

    waste = N % (proccess - 1);
    n = N / (proccess - 1);

    if (rank == 0)
    {
        FillMatrix(matrixA, N);
        FillMatrix(matrixB, N);
    }

    MPI_Bcast(matrixA, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(matrixB, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        processN = n;

        for (i = 1; i <= (proccess - 1); i++)
        {
            if (i == proccess - 1)
                processN = waste + n;
            MPI_Recv(matrixC + ((i - 1) * N * n), processN * N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        PrintMatrix(matrixA, N, N, "Matrix A");
        PrintMatrix(matrixB, N, N, "Matrix B");
        PrintMatrix(matrixC, N, N, "Matrix C (result)");
    }
    else
    {
        if (rank == proccess - 1)
            processN = waste + n;
        else
            processN = n;

        Multiply(matrixA, matrixB, matrixC, processN, N);
        MPI_Send(matrixC, processN * N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}

void PrintMatrix(double *matrix, int row, int column, char name[])
{
    int i;
    int j;

    printf("%s\n", name);

    for (i = 0; i < row; i++)
    {
        for (j = 0; j < column; j++)
            printf("%.2le\t", matrix[(i * column) + j]);
        printf("\n");
    }
}

void FillMatrix(double *matrix, int N)
{
    int i;
    int j;

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            matrix[(i * N) + j] = a;
}

void Multiply(double *matrixA, double *matrixB, double *matrixC, int NX, int NY)
{
    int i;
    int j;
    int k;
    double result;

    for (i = 0; i < NX; i++)
        for (j = 0; j < NY; j++)
        {
            result = 0.0;
            for (k = 0; k < NY; k++)
                result += matrixA[(i * NY) + k] * matrixB[(k * NY) + j];
            matrixC[(i * NY) + j] = result;
        }
}
