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
 * Run: mpiexec -np 10 ./matmult-mpi-c.o
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

void FillMatrix(float *array, int size);
void PrintArray(float *array, int row, int column, char name[]);
void Multiply(float *matrixA, float *matrixB, float *matrixC, int sizeX, int sizeY);
float MultiplyArray(float *matrixA, float *matrixB, int size, int x, int y);
void Init(int *waste, int *n, int size, int *sum, int proccess);

int main(int argc, char *argv[])
{
    int rank, proccess, size = 2;
    int i, j, k, waste, n, processSize;
    float matrixA[size * size], matrixB[size * size], matrixC[size * size];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proccess);

    Init(&waste, &n, size, &i, proccess);

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
        PrintArray(matrixA, size, size, "matrixA");
        PrintArray(matrixB, size, size, "matrixB");
        PrintArray(matrixC, size, size, "matrixC");
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

    MPI_Finalize();
    return 0;
}

/**
* Llena una matriz cuadrada con valores aleatorios.
* array - Matriz cuadrada.
* size - Tamaño de la matriz.
*/
void FillMatrix(float *array, int size)
{
    int i, j;

    for (i = 0; i < size; i++)
        for (j = 0; j < size; j++)
            *(array + (i * size) + j) = rand() % (11 * (i + 1)) * 1.12;
}

/**
* Imprime una matriz con un mensaje y un identificador.
* array - Matriz.
* row - Numero de renglones de la matriz.
* column - Numero de columnas de la matriz.
* name - Mensaje para imprimir.
* id - Identificador del proceso.
*/
void PrintArray(float *array, int row, int column, char name[])
{
    int i, j;

    printf("%s\n", name);

    for (i = 0; i < row; i++)
    {
        for (j = 0; j < column; j++)
            printf("%.2f\t", *(array + (i * column) + j));
        printf("\n");
    }
}

/**
* Multiplica dos matrices cuadradas A, B y almacena el resultado en una tercera matriz C.
* matrixA - Primera matriz cuadrada.
* matrixB - Segunda matriz cuadrada.
* matrixC - Matriz resultado.
* size - Tamaño de las matrices.
*/
void Multiply(float *matrixA, float *matrixB, float *matrixC, int sizeX, int sizeY)
{
    int i, j;

    for (i = 0; i < sizeX; i++)
        for (j = 0; j < sizeY; j++)
            *(matrixC + (i * sizeX) + j) = MultiplyArray(matrixA, matrixB, sizeY, i, j);
}

/**
* Multiplica el renglon x de la matriz A por la columna y de la matriz B 
* y almacena el resultado una variable de regreso.
* matrixA - Primera matriz.
* matrixB - Segunda matriz.
* size - Tamaño de los arreglos.
* x - Renglon seleccionado de la matriz A. 
* y - Columna seleccionada de la matriz B.
* Forma de acceder a las posiciones de la matriz: matriz + renglon * tamaño + columna
*/
float MultiplyArray(float *matrixA, float *matrixB, int size, int x, int y)
{
    int i;
    float r = 0.0;

    for (i = 0; i < size; i++)
        r += *(matrixA + (x * size) + i) * *(matrixB + (i * size) + y);

    return r;
}

/**
* Inicializa el tamaño de informacion que debe repartir el primer proceso
* con todos los demas.
* waste - Residuo para el ultimo proceso.
* n -Tamaño de la informacion que se debe repartir entre los procesos.
* size -Tamaño de la informacion total.
* sum -Variable entera para inicializar en 0.
* process - Numero de procesos.
*/
void Init(int *waste, int *n, int size, int *sum, int proccess)
{
    *waste = size % (proccess - 1);
    *n = (size - *waste) / (proccess - 1);
    *sum = 0;
}