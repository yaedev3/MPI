#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void PrintMatrix(double *matrix, int size, char name[]);
void FillMatrix(double *matrixA, double *matrixB, int size);
void Multiply(double *matrixA, double *matrixB, double *matrixC, int size);

void main()
{
    int size;
    double *matrixA;
    double *matrixB;
    double *matrixC;

    size = 5;
    matrixA = (double *)malloc(sizeof(double) * size * size);
    matrixB = (double *)malloc(sizeof(double) * size * size);
    matrixC = (double *)malloc(sizeof(double) * size * size);

    FillMatrix(matrixA, matrixB, size);

    Multiply(matrixA, matrixB, matrixC, size);

    PrintMatrix(matrixA, size, "Matrix A");
    PrintMatrix(matrixB, size, "Matrix B");
    PrintMatrix(matrixC, size, "Matrix C (result)");

    free(matrixA);
    free(matrixB);
    free(matrixC);
}

void PrintMatrix(double *matrix, int size, char name[])
{
    int i;
    int j;

    printf("%s\n", name);

    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
            printf("%.15le\t", matrix[(i * size) + j]);
        printf("\n");
    }
}

void FillMatrix(double *matrixA, double *matrixB, int size)
{
    int i;
    int j;

    for (i = 0; i < size; i++)
        for (j = 0; j < size; j++)
        {
            matrixA[(i * size) + j] = 1.0E-10;
            matrixB[(i * size) + j] = 1.0E-10;
        }
}

void Multiply(double *matrixA, double *matrixB, double *matrixC, int size)
{
    int i;
    int j;
    int k;
    double result;

    for (i = 0; i < size; i++)
        for (j = 0; j < size; j++)
        {
            result = 0.0;
            for (k = 0; k < size; k++)
                result += matrixA[(i * size) + k] * matrixB[(k * size) + j];
            matrixC[(i * size) + j] = result;
        }
}
