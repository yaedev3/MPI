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

void PrintMatrix(float *matrix, int size, char name[]);
void FillMatrix(float *matrix, int size);
void Multiply(float *matrixA, float *matrixB, float *matrixC, int size);

void main()
{
    int size;
    float *matrixA;
    float *matrixB;
    float *matrixC;

    size = 5;
    matrixA = (float *)malloc(sizeof(float) * size * size);
    matrixB = (float *)malloc(sizeof(float) * size * size);
    matrixC = (float *)malloc(sizeof(float) * size * size);

    FillMatrix(matrixA, size);
    FillMatrix(matrixB, size);

    Multiply(matrixA, matrixB, matrixC, size);

    PrintMatrix(matrixA, size, "Matrix A");
    PrintMatrix(matrixB, size, "Matrix B");
    PrintMatrix(matrixC, size, "Matrix C (result)");

    free(matrixA);
    free(matrixB);
    free(matrixC);
}

void PrintMatrix(float *matrix, int size, char name[])
{
    int i;
    int j;

    printf("%s\n", name);

    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
            printf("%.2f\t", matrix[(i * size) + j]);
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

void Multiply(float *matrixA, float *matrixB, float *matrixC, int size)
{
    int i;
    int j;
    int k;
    float result;

    for (i = 0; i < size; i++)
        for (j = 0; j < size; j++)
        {
            result = 0.0;
            for (k = 0; k < size; k++)
                result += matrixA[(i * size) + k] * matrixB[(k * size) + j];
            matrixC[(i * size) + j] = result;
        }
}