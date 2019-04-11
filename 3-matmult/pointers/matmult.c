#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Valor constante para llenar la matriz
static double a = 1.0E-10;

void FillMatrix(double *matrixA, double *matrixB, int N);
void Multiply(double *matrixA, double *matrixB, double *matrixC, int N);
double AddMatrix(double *matrix, int N);
void OpenFile(int *N);
void SaveFile(struct tm *start, struct tm *end, double error, double elapsed, int N);

int main(int argc, char *argv[])
{
    int N;               // Dimension de la matriz
    double *matrixA;     // Primera matriz
    double *matrixB;     // Segunda matriz
    double *matrixC;     // Matriz resultado
    double result;       // Resultado de la suma de la matriz resultado
    double estimation;   // Estimacion del calculo
    double error;        // Error encontrado
    double elapsed;      // Tiempo que tomo ejecutarse el programa
    time_t t;            // Variable de tiempo
    struct tm *start;    // Hora de inicio
    struct tm *end;      // Hora de termino
    clock_t start_clock; // Tiempo de inicio
    clock_t stop_clock;  // Tiempo de termino

    // Inicia la variable de tiempo
    t = time(NULL);

    // Establece la hora de inicio
    start = localtime(&t);
    start_clock = clock();

    // Asigna la dimension de la matriz
    OpenFile(&N);

    // Reserva la memoria para las tres matrices
    matrixA = (double *)malloc(sizeof(double) * N * N);
    matrixB = (double *)malloc(sizeof(double) * N * N);
    matrixC = (double *)malloc(sizeof(double) * N * N);

    // Llena la matriz A y B con el valor constante
    FillMatrix(matrixA, matrixB, N);

    // Multiplica las matrices A y B guardando el valor en la matriz C
    Multiply(matrixA, matrixB, matrixC, N);

    //Calcula la suma de los valores de la matriz C
    result = AddMatrix(matrixC, N);

    // Calculo estimado con la formula a^2*N^3
    estimation = pow(N, 3) * pow(a, 2);

    // Calcula el porcentaje de error.
    error = fabs(result - estimation) / estimation * 100.0;

    // Libera la memoria de las tres matrices
    free(matrixA);
    free(matrixB);
    free(matrixC);

    // Establece la hora en que termino de calcular
    end = localtime(&t);
    stop_clock = clock();

    // Calcula el tiempo que tomo ejecutar el programa
    elapsed = (double)(stop_clock - start_clock) / CLOCKS_PER_SEC;

    // Guarda el resultado en un archivo
    SaveFile(start, end, error, elapsed, N);

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
    int N            // Dimension de la matriz
)
{
    int i;         // Indice del renglon
    int j;         // Indice de la columna
    int k;         // Indice de la multiplicacion
    double result; // Resultado de la multiplicacion

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
        {
            result = 0.0;
            for (k = 0; k < N; k++)
                result += matrixA[(i * N) + k] * matrixB[(k * N) + j];
            matrixC[(i * N) + j] = result;
        }
}

// Suma todos los elementos de una matriz y regresa el resultado
double AddMatrix(
    double *matrix, // Matriz resultado
    int N           // Dimension de la matriz
)
{
    double result; // Resultado de la suma
    int i;         // Indice del renglon
    int j;         // Indice de la columna

    result = 0.0;

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            result += matrix[(i * N) + j];

    return result;
}

// Abre un archivo con la dimension de la matriz
void OpenFile(
    int *N // Dimension de la matriz
)
{
    FILE *file;

    file = fopen("parameters.dat", "r");

    if (file == NULL)
        printf("No se puede abrir el archivo.\n");
    else
    {
        fscanf(file, "%d", N);
        fclose(file);
    }
}

// Crea un archivo de salida con la hora de inicio, de termino y el tiempo que tomo correr el programa
// Asi como el porcentaje de error
void SaveFile(
    struct tm *start, // Hora de inicio
    struct tm *end,   // Hora de termino
    double error,     // Porcentaje de error
    double elapsed,   // Tiempo que paso
    int N             // Dimension de la matriz
)
{
    FILE *file;
    char file_name[64];
    char output[50];

    sprintf(file_name, "serial-c-%d-%d-%d-%d-%d.txt", N, end->tm_mday, end->tm_hour, end->tm_min, end->tm_sec);

    file = fopen(file_name, "w+");

    strftime(output, sizeof(output), "%c", start);
    fprintf(file, "Hora de inicio\n%s\n", output);

    strftime(output, sizeof(output), "%c", end);
    fprintf(file, "Hora de termino\n%s\n", output);

    fprintf(file, "Tiempo de ejecucion\n%.15lf\n", elapsed);

    fprintf(file, "Error\n%.15le\n", error);

    fclose(file);
}
