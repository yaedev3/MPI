#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <math.h>

static double a = 1.0E-10;

void FillMatrix(double *matrixA, double *matrixB, int N);
void Multiply(double *matrixA, double *matrixB, double *matrixC, int Nx, int Ny);
double AddMatrix(double *matrix, int Nx, int Ny);

int main(int argc, char *argv[])
{
    int N;
    int rank;
    int proccess;
    int waste;
    int n_local;
    int processSize;
    int i;
    double *matrixA;
    double *matrixB;
    double *matrixC;
    double result;
    double result_local;
    double estimation;
    double error;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proccess);

    // Verifica si tiene los argumentos necesarios para inicializa el tamaño de las matrices
    if (argc < 2)
    {
        printf("Falta el argumento del tamaño\n");
        return -1;
    }

    // Asigna el valor del primer argumento a la variable de tamaño
    sscanf(argv[1], "%d", &N);

    // Calcula el sobrante de datos en caso de que los procesos no sean
    // multiplos de la informacion
    waste = N % (proccess - 1);

    // Calcula el numero de datos que va a tomar cada proceso
    n_local = N / (proccess - 1);

    //Inicializa la matriz B para todos los procesos
    matrixB = (double *)malloc(sizeof(double) * N * N);

    if (rank == 0)
    {
        // Inicializa matriz A con el tamaño total (N * N)
        matrixA = (double *)malloc(sizeof(double) * N * N);

        // Inicializa el resultado en 0.0
        result = 0.0;

        // Llena la matriz A y B con el valor constante
        FillMatrix(matrixA, matrixB, N);
    }

    // Comparte la informacion de la matriz B con los demas procesos.
    MPI_Bcast(matrixB, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        // Envia partes de la matriz A a todos los demas procesos
        processSize = n_local;
        for (i = 1; i <= proccess - 1; i++)
        {
            if (i == proccess - 1)
                processSize = waste + n_local;
            MPI_Send(matrixA + ((i - 1) * N * n_local), processSize * N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }

        // Recibe el resultado de la suma de todos los elementos de la matriz C
        for (i = 1; i <= proccess - 1; i++)
        {
            MPI_Recv(&result_local, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            result += result_local;
        }

        // Calculo estimado con la formula a^2*N^3.
        estimation = pow(N, 3) * pow(a, 2);

        // Calcula el % de error.
        error = fabs(result - estimation) / estimation * 100.0;

        // Imprime el % de error.
        printf("Error %le N = %d\n", error, N);
    }
    else
    {
        // Verifica si es el ultimo proceso para calcular el reciduo
        if (rank == proccess - 1)
            processSize = waste + n_local;
        else
            processSize = n_local;

        // Inicializa las matrices A y C con los tamaños correspondientes a cada proceso.
        matrixA = (double *)malloc(sizeof(double) * processSize * N);
        matrixC = (double *)malloc(sizeof(double) * processSize * N);

        // Recibe la matriz A del proceso 0
        MPI_Recv(matrixA, processSize * N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Hace la multiplicacion de matrices.
        Multiply(matrixA, matrixB, matrixC, processSize, N);

        // Hace la suma de todos los elementos de la matriz C
        result_local = AddMatrix(matrixC, processSize, N);

        // Envia el resultado de la multiplicacion al proceso 0.
        MPI_Send(&result_local, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);

        //Libera la memoria de la matriz C
        free(matrixC);
    }

    //Libera la memoria de las matrices A y B
    free(matrixA);
    free(matrixB);

    MPI_Finalize();
    return 0;
}

void FillMatrix(double *matrixA, double *matrixB, int N)
{
    int i;
    int j;

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
        {
            matrixA[(i * N) + j] = a;
            matrixB[(i * N) + j] = a;
        }
}

void Multiply(double *matrixA, double *matrixB, double *matrixC, int Nx, int Ny)
{
    int i;
    int j;
    int k;
    double result;

    for (i = 0; i < Nx; i++)
        for (j = 0; j < Ny; j++)
        {
            result = 0.0;
            for (k = 0; k < Ny; k++)
                result += matrixA[(i * Ny) + k] * matrixB[(k * Ny) + j];
            matrixC[(i * Ny) + j] = result;
        }
}

double AddMatrix(double *matrix, int Nx, int Ny)
{
    double result;
    int i;
    int j;

    result = 0.0;

    for (i = 0; i < Nx; i++)
        for (j = 0; j < Ny; j++)
            result += matrix[(i * Nx) + j];

    return result;
}
