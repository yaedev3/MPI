#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <math.h>

// Valor constante para llenar la matriz
static double a = 1.0E-10;

void FillMatrix(double *matrixA, double *matrixB, int N);
void Multiply(double *matrixA, double *matrixB, double *matrixC, int Nx, int Ny);
double AddMatrix(double *matrix, int Nx, int Ny);

int main(int argc, char *argv[])
{
    int N;               // Dimension de la matriz
    int rank;            // Indice de cada proceso
    int proccess;        // Numero total de procesos
    int waste;           // Residuo de informacion
    int n_local;         // Tamaño de informacion por proceso
    int processSize;     // Tamaño corregido
    int i;               // Iterador de procesos
    double *matrixA;     // Primera matriz
    double *matrixB;     // Segunda matriz
    double *matrixC;     // Matriz resultado
    double result;       // Resultado de la suma de la matriz resultado
    double result_local; // Resultado de la suma local de cada proceso
    double estimation;   // Estimacion del calculo
    double error;        // Error encontrado

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proccess);

    // Verifica si tiene los argumentos necesarios para inicializa el tamaño de las matrices
    if (argc < 2)
    {
        printf("Falta el argumento del tamaño\n");
        return -1;
    }

    // Asigna la dimension de la matriz
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
        printf("Error %.15le N = %d\n", error, N);
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

    // Libera la memoria de las matrices A y B
    free(matrixA);
    free(matrixB);

    // Termina MPI
    MPI_Finalize();
    
    return 0;
}

// Llena las dos matrices con el valor constante
void FillMatrix(
    double *matrixA, // Primera matriz
    double *matrixB, // Segunda matriz
    int N            // Dimension de la matriz
)
{
    int i; // Indice el renglon
    int j; // Indice de la columna

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
        {
            matrixA[(i * N) + j] = a;
            matrixB[(i * N) + j] = a;
        }
}

// Multiplica las dos matrices y almacena el resultado en la matriz de resultado
void Multiply(
    double *matrixA, // Primera matriz
    double *matrixB, // Segunda matriz
    double *matrixC, // Matriz resultado
    int Nx,          // Tamaño de renglones
    int Ny           // Tamaño de columnas
)
{
    int i;         // Indice del renglon
    int j;         // Indice de la columna
    int k;         // Indice de la multiplicacion
    double result; // Resultado de la multiplicacion

    for (i = 0; i < Nx; i++)
        for (j = 0; j < Ny; j++)
        {
            result = 0.0;
            for (k = 0; k < Ny; k++)
                result += matrixA[(i * Ny) + k] * matrixB[(k * Ny) + j];
            matrixC[(i * Ny) + j] = result;
        }
}

// Suma todos los elementos de una matriz y regresa el resultado
double AddMatrix(
    double *matrix, // Matriz resultado
    int Nx,         // Tamaño de renglones
    int Ny          // Tamaño de columnas
)
{
    double result; // Resultado de la suma
    int i;         // Indice del renglon
    int j;         // Indice de la columna

    result = 0.0;

    for (i = 0; i < Nx; i++)
        for (j = 0; j < Ny; j++)
            result += matrix[(i * Nx) + j];

    return result;
}