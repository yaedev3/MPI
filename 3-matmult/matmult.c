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

void PrintMatrix(float *array, int size, char name[]);
void FillMatrix(float *array, int size);
void Multiply(float *matrixA, float *matrixB, float *matrixC, int size);

void main()
{
	int size = 5;
	float matrixA[size * size], matrixB[size * size], matrixC[size * size];

	FillMatrix(matrixA, size);
	FillMatrix(matrixB, size);

	PrintMatrix(matrixA, size, "matrixA");
	PrintMatrix(matrixB, size, "matrixB");
	Multiply(matrixA, matrixB, matrixC, size);
	PrintMatrix(matrixC, size, "matrixC");
}

void PrintMatrix(float *array, int size, char name[])
{
	int i, j;

	printf("Print Array %s\n", name);

	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
			printf("%.2f\t", *(array + (i * size) + j));
		printf("\n");
	}
}

void FillMatrix(float *array, int size)
{
	int i, j;

	for (i = 0; i < size; i++)
		for (j = 0; j < size; j++)
			*(array + (i * size) + j) = rand() % (11 * (i + 1)) * 1.12;
}

void Multiply(float *matrixA, float *matrixB, float *matrixC, int size)
{
	int i, j, k;
	float result;

	for (i = 0; i < size; i++)
		for (j = 0; j < size; j++)
		{
			result = 0.0;
			for (k = 0; k < size; k++)
				result += *(matrixA + (i * size) + k) * *(matrixB + (k * size) + j);
			*(matrixC + (i * size) + j) = result;
		}
}