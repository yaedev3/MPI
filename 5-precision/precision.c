#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static double a = 1.0E-10;

void FillMatrix(double *matrixA, double *matrixB, int N);
void Multiply(double *matrixA, double *matrixB, double *matrixC, int N);
double AddMatrix(double *matrix, int N);

int main(int argc, char *argv[])
{
    int N;                      //
    double *matrixA;            //
    double *matrixB;            //
    double *matrixC;            //
    double result;              //
    double estimation;          //
    double error;               //

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
    FillMatrix(matrixA, matrixB, N);

    // Multiplica las matrices A y B guardando el valor en la matriz C
    Multiply(matrixA, matrixB, matrixC, N);

    //Calcula la suma de los valores de la matriz C.
    result = AddMatrix(matrixC, N);

    // Calculo estimado con la formula a^2*N^3.
    estimation = pow(N, 3) * pow(a, 2);

    // Calcula el % de error.
    error = fabs(result - estimation) / estimation * 100.0;

    // Imprime el % de error.
    printf("Error %.15le N = %d\n", error, N);

    // Libera la memoria de las tres matrices
    free(matrixA);
    free(matrixB);
    free(matrixC);
    
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

double AddMatrix(double *matrix, int N)
{
    double result;
    int i;
    int j;

    result = 0.0;

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            result += matrix[(i * N) + j];

    return result;
}
