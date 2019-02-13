#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static double a = 1.0E-10;

void FillMatrix(double *matrixA, double *matrixB, int size);
void Multiply(double *matrixA, double *matrixB, double *matrixC, int size);
double AddMatrix(double *matrix, int size);

void main(int argc, char *argv[])
{
    int size;
    double *matrixA;
    double *matrixB;
    double *matrixC;
    double result;

    printf("%d %d\n", argv[0], argc);

    size = 5;
    matrixA = (double *)malloc(sizeof(double) * size * size);
    matrixB = (double *)malloc(sizeof(double) * size * size);
    matrixC = (double *)malloc(sizeof(double) * size * size);

    FillMatrix(matrixA, matrixB, size);
    Multiply(matrixA, matrixB, matrixC, size);

    result = AddMatrix(matrixC, size);
    printf("result calc: %.15le\n", size * size * size * a * a);
    printf("result: %.15le\n", result);

    free(matrixA);
    free(matrixB);
    free(matrixC);
}

void FillMatrix(double *matrixA, double *matrixB, int size)
{
    int i;
    int j;

    for (i = 0; i < size; i++)
        for (j = 0; j < size; j++)
        {
            matrixA[(i * size) + j] = a;
            matrixB[(i * size) + j] = a;
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

double AddMatrix(double *matrix, int size)
{
    double result;
    int i;
    int j;

    result = 0.0;

    for (i = 0; i < size; i++)
        for (j = 0; j < size; j++)
            result += matrix[(i * size) + j];

    return result;
}