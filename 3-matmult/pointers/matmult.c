#include <stdio.h>
#include <stdlib.h>

// Valor constante para llenar la matriz

static double a = 1.0E-10;

void PrintMatrix(double *matrix, int N, char name[]);
void FillMatrix(double *matrix, int N);
void Multiply(double *matrixA, double *matrixB, double *matrixC, int N);

int main(int argc, char *argv[])
{
    int N;                  // 
    double *matrixA;        //
    double *matrixB;        //
    double *matrixC;        //

    // Verifica si tiene los argumentos necesarios para inicializa el tamaño de las matrices
    if (argc < 2)
    {
        printf("Falta el argumento del tamaño\n");
        return -1;
    }

    // Asigna el valor del primer argumento a la variable de tamaño
    sscanf(argv[1], "%d", &N);

    // Reserva la memoria para las tres matrices
    matrixA = (double *)malloc(sizeof(double) * N * N);
    matrixB = (double *)malloc(sizeof(double) * N * N);
    matrixC = (double *)malloc(sizeof(double) * N * N);

    // Llena la matriz A y B con el valor constante
    FillMatrix(matrixA, N);
    FillMatrix(matrixB, N);

    // Multiplica las matrices A y B guardando el valor en la matriz C
    Multiply(matrixA, matrixB, matrixC, N);

    // Imprime el contenido de las tres matrices
    PrintMatrix(matrixA, N, "Matrix A");
    PrintMatrix(matrixB, N, "Matrix B");
    PrintMatrix(matrixC, N, "Matrix C (result)");

    // Libera la memoria de las tres matrices
    free(matrixA);
    free(matrixB);
    free(matrixC);

    return 0;
}

/**
 * @brief 
 * 
 * @param matrix 
 * @param N 
 * @param name 
 */
void PrintMatrix(double *matrix, int N, char name[])
{
    int i;
    int j;

    printf("%s\n", name);

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
            printf("%.2le\t", matrix[(i * N) + j]);
        printf("\n");
    }
}

/**
 * @brief 
 * 
 * @param matrix 
 * @param N 
 */
void FillMatrix(double *matrix, int N)
{
    int i;
    int j;

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            matrix[(i * N) + j] = a;
}

/**
 * @brief 
 * 
 * @param matrixA 
 * @param matrixB 
 * @param matrixC 
 * @param N 
 */
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
