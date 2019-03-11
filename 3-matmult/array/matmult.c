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

void PrintMatrix(double *matrix, int size, char name[]);
void FillMatrix(double *matrix, int size);
void Multiply(double *matrixA, double *matrixB, double *matrixC, int size);

void main()
{
	int size = 5;
	double matrixA[size * size];
	double matrixB[size * size];
	double matrixC[size * size];

	FillMatrix(matrixA, size);
	FillMatrix(matrixB, size);

	Multiply(matrixA, matrixB, matrixC, size);

	PrintMatrix(matrixA, size, "Matrix A");
	PrintMatrix(matrixB, size, "Matrix B");
	PrintMatrix(matrixC, size, "Matrix C (result)");
}

void PrintMatrix(double *matrix, int size, char name[])
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

void FillMatrix(double *matrix, int size)
{
	int i;
	int j;

	for (i = 0; i < size; i++)
		for (j = 0; j < size; j++)
			matrix[(i * size) + j] = 1.0;
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
