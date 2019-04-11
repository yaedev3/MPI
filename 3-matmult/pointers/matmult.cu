#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>

static double a = 1.0E-10;

void FillMatrix(double *matrixA, double *matrixB, int size);
void OpenFile(int *N);
void SaveFile(struct tm *start, struct tm *end, double error, double elapsed, int N);

__global__ void Multiply(double* A, double* B, double* C, int N)
{
    double result;      // Acumula la suma del renglon por la columna 
    int index;          // Indice del vector 
	int ix;             // Indica el renglon 
	int iy;             // Toma valores solo entre 0 a N-1
    int k;              // Iterador 
    
    index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index < N * N)
	{
		ix = index / N;
		iy = index % N;
		result = 0.0;

		for(k = 0; k < N; k++)
			result += A[k + N * ix] * B[k * N + iy ];

		C[iy + N * ix] = result;
	}
}

__global__ void AddMatrix(double* C, int N, double* result)
{
    long i;
    long j;

    for(i = 0; i < N; i++)
        for(j = 0; j < N; j++)
            *result += C[i + N * j];
}

int main(int argc, char *argv[])
{
    //Variables 
	int N;                  // Tamaño de la matriz cuadrada.
    size_t size;            // Tamaño total en memoria.
    size_t sizeDouble;      // Tamaño en memoria del tipo doble.
	double* h_matrixA;      // Matriz A en el equipo.
	double* h_matrixB;      // Matriz B en el equipo.
	double* h_matrixC;      // Matriz C (resultado) en el equipo.
	double* d_matrixA;      // Matriz A en la memoria de la GPU.
	double* d_matrixB;      // Matriz B en la memoria de la GPU.
	double* d_matrixC;      // Matriz C (resultado) en la memoria de la GPU.
    double* h_result;       // Sumatoria de los valores de la multiplicacion de matrices en el equipo.
    double* d_result;       // Sumatoria de los valores de la multiplicacion de matrices en la GPU.
    int Tam;                // Numero de datos que se manejan
	int threads;            // Hilos por bloque 
    int blocks;             // Numero de bloques necesario para procesar los datos 
    double estimation;      // Estimacion del calculo
    double error;           // Error encontrado
    double elapsed;         // Tiempo que tomo ejecutarse el programa
    time_t t;               // Variable de tiempo
    struct tm *start;       // Hora de inicio
    struct tm *end;         // Hora de termino
    clock_t start_clock;    // Tiempo de inicio
    clock_t stop_clock;     // Tiempo de termino

    // Inicia la variable de tiempo
    t = time(NULL);

    // Establece la hora de inicio
    start = localtime(&t);
    start_clock = clock();

    // Asigna la dimension de la matriz
    OpenFile(&N);

    // Establece el tamaño total de la matriz en memoria
    size = N * sizeof(double) * N;

    // Establecec el tamaño del tipo de dato double
    sizeDouble = sizeof(double);
    
    // Asigna el numero de hilos y calcula el numero de bloques
    Tam = N * N;
	cudaDeviceGetAttribute(&threads, cudaDevAttrMaxThreadsPerBlock, 0);
    blocks = Tam / threads;
    
    if(Tam % threads > 0) //Si sobran datos, aumenta los bloques en 1
        blocks++;

	//En la memoria del equipo 
	h_matrixA = (double*)malloc(size);
	h_matrixB = (double*)malloc(size);
    h_matrixC = (double*)malloc(size);
    h_result = (double*)malloc(sizeDouble);
    
	//En la memoria de la GPU
	cudaMalloc(&d_matrixA, size);
	cudaMalloc(&d_matrixB, size);
    cudaMalloc(&d_matrixC, size);
    cudaMalloc(&d_result, sizeDouble);
    
    // Llena las matrices h_matrixA y h_matrixB
    FillMatrix(h_matrixA, h_matrixB, N);

    // Copia los arreglos de memoria del CPU a memoria de la GPU 
	cudaMemcpy(d_matrixA, h_matrixA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixB, h_matrixB, size, cudaMemcpyHostToDevice);

    // Mandar llamar la multiplicacion de matrices.
    Multiply<<<blocks, threads >>>(d_matrixA, d_matrixB, d_matrixC, N);

    // Inicializa la variable d_result con 0.0 y lo copia a la memoria de la GPU
    *h_result = 0.0;
    cudaMemcpy(d_result, h_result, sizeDouble, cudaMemcpyHostToDevice);

    // Suma los valores de la multiplicacion de matrices
    AddMatrix<<<1, 1>>>(d_matrixC, N, d_result);

    //Copia el resultado de la suma de los elementos de la matriz en la memoria
	cudaMemcpy(h_result, d_result, sizeDouble, cudaMemcpyDeviceToHost);

    // Calculo estimado con la formula a^2*N^3.
    estimation = pow(N, 3) * pow(a, 2);

    // Calcula el % de error.
    error = fabs(*h_result - estimation) / estimation * 100.0;
    
	// Libera espacio del equipo
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
    cudaFree(d_matrixC);
    cudaFree(d_result);

	// Libera espacio de la tarjeta de video
	free(h_matrixA);
	free(h_matrixB);
    free(h_matrixC);
    free(h_result);

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

    sprintf(file_name, "cuda-%d-%d-%d-%d-%d.txt", N, end->tm_mday, end->tm_hour, end->tm_min, end->tm_sec);

    file = fopen(file_name, "w+");

    strftime(output, sizeof(output), "%c", start);
    fprintf(file, "Hora de inicio\n%s\n", output);

    strftime(output, sizeof(output), "%c", end);
    fprintf(file, "Hora de termino\n%s\n", output);

    fprintf(file, "Tiempo de ejecucion\n%.15lf\n", elapsed);

    fprintf(file, "Error\n%.15le\n", error);

    fclose(file);
}
