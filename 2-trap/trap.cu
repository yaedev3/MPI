#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

void OpenFile(double *a, double *b, long *n);
void SaveFile(struct tm *start, struct tm *end, double error, double elapsed, long N);

__global__ void Trap(double a, double b, long n, double h, double* integral)
{
    long k;

    *integral = (a * a + b * b) / 2.0;

    for (k = 1; k <= n - 1; k++)
        *integral += (a + k * h) * (a + k * h);

    *integral = *integral * h;
}

int main(int argc, char *argv[])
{
    size_t sizeDouble;      // Tamaño en memoria del tipo doble
    int threads;            // Hilos por bloque 
    int blocks;             // Numero de bloques necesario para procesar los datos 
    double h_integral;     /* Store result in integral */
    double* d_integral;     /* Store result in integral */
    double a;            /* Left endpoints */
    double b;            /* Right endpoints */
    double h;            /* Height of trapezoids */
    long n;              /* Number of trapezoids */
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

    //
    OpenFile(&a, &b, &n);

    // Establece el tamaño del tipo de dato double
    sizeDouble = sizeof(double);
    
    // Asigna el numero de hilos y calcula el numero de bloques
	cudaDeviceGetAttribute(&threads, cudaDevAttrMaxThreadsPerBlock, 0);
    blocks = n / threads;

    if(n % threads > 0) //Si sobran datos, aumenta los bloques en 1
        blocks++;

    cudaMalloc(&d_integral, sizeDouble);

    h = (b - a) / n;

    // Mandar llamar el kernel
    Trap<<<1, 1 >>>(a, b, n, h, d_integral);
    cudaMemcpy(&h_integral, d_integral, sizeDouble, cudaMemcpyDeviceToHost);

    // Estimacion del resultado
    estimation = 9.0;

    // Calcula el porcentaje de error.
    error = fabs(h_integral - estimation) / estimation * 100.0;

    // Establece la hora en que termino de calcular
    end = localtime(&t);
    stop_clock = clock();

    // Calcula el tiempo que tomo ejecutar el programa
    elapsed = (double)(stop_clock - start_clock) / CLOCKS_PER_SEC;

    // Libera espacio en la tarjeta de memoria
    cudaFree(d_integral);

    // Guarda el resultado en un archivo
    SaveFile(start, end, error, elapsed, n);

    return 0;
}


void OpenFile(double *a, double *b, long *n)
{
    FILE *file;
    char *input_file;

    input_file = "parameters.dat";

    file = fopen(input_file, "r");

    fscanf(file, "%lf %lf %ld", a, b, n);

    fclose(file);
}

// Crea un archivo de salida con la hora de inicio, de termino y el tiempo que tomo correr el programa
// Asi como el porcentaje de error
void SaveFile(
    struct tm *start, // Hora de inicio
    struct tm *end,   // Hora de termino
    double error,     // Porcentaje de error
    double elapsed,   // Tiempo que paso
    long N            // Dimension de la matriz
)
{
    FILE *file;
    char file_name[64];
    char output[50];

    sprintf(file_name, "cuda-%ld-%d-%d-%d-%d.txt", N, end->tm_mday, end->tm_hour, end->tm_min, end->tm_sec);

    file = fopen(file_name, "w+");

    strftime(output, sizeof(output), "%c", start);
    fprintf(file, "Hora de inicio\n%s\n", output);

    strftime(output, sizeof(output), "%c", end);
    fprintf(file, "Hora de termino\n%s\n", output);

    fprintf(file, "Tiempo de ejecucion\n%.15lf\n", elapsed);

    fprintf(file, "Error\n%.15le\n", error);

    fclose(file);
}
