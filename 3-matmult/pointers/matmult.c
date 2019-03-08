/* File: matmult.c
 *
 * Purpose: 
 * 
 * Input:
 * 
 * Output:
 * 
 * Compile: gcc -o matmult-c.o matmult.c
 * 
 * Run: ./matmult-c.o
 * 
 * Algorithm:
 * 
 * Note:
 * 
 * */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void PrintMatrix(double *matrix, int N, char name[]);
void FillMatrix(double *matrix, int N);
void Multiply(double *matrixA, double *matrixB, double *matrixC, int N);

void main()
{
    int N;
    double *matrixA;
    double *matrixB;
    double *matrixC;

    N = 5;
    matrixA = (double *)malloc(sizeof(double) * N * N);
    matrixB = (double *)malloc(sizeof(double) * N * N);
    matrixC = (double *)malloc(sizeof(double) * N * N);

    FillMatrix(matrixA, N);
    FillMatrix(matrixB, N);

    Multiply(matrixA, matrixB, matrixC, N);

    PrintMatrix(matrixA, N, "Matrix A");
    PrintMatrix(matrixB, N, "Matrix B");
    PrintMatrix(matrixC, N, "Matrix C (result)");

    free(matrixA);
    free(matrixB);
    free(matrixC);
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

void FillMatrix(double *matrix, int N)
{
    int i;
    int j;

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            matrix[(i * N) + j] = rand() % (11 * (i + 1)) * 1.12;
}

void Multiply(double *matrixA, double *matrixB, double *matrixC, int N)
{
    int i;
    int j;
    int k;
    double result;

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
        {
            result = 0.0;
            for (k = 0; k < N; k++)
                result += matrixA[(i * N) + k] * matrixB[(k * N) + j];
            matrixC[(i * N) + j] = result;
        }
}
