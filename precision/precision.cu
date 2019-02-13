#include <stdio.h>
#include <cuda_runtime.h>

static double a = 1.0E-10;

void FillMatrix(double *matrixA, double *matrixB, int size);

__global__ void Multiply(double* A, double* B, double* C, int N)
{
    double result;      //Acumula la suma del renglon por la columna 
    int index;          //indice del vector 
	int ix;             //ix indica el renglon 
	int iy;             //iy toma valores solo entre 0 a N-1
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
    int index;          //indice del vector 
	int ix;             //ix indica el renglon 
	int iy;             //iy toma valores solo entre 0 a N-1
    
    index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index < N * N)
	{
		ix = index / N;
		iy = index % N;
        *result += C[iy + N * ix];
	}
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

    //printf("%d %d\n", argv[0], argc);

    //Asignacion de variables
	N = 5;
    size = N * sizeof(double) * N;
    sizeDouble = sizeof(double);
    Tam = N * N;
	threads = 1024;
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

    // Inicializa la variable d_result con 0.0 y lo copia a la memoria de la GPU
    h_result[0] = 0.0;
    cudaMemcpy(d_result, h_result, sizeDouble, cudaMemcpyHostToDevice);

    // Mandar llamar la multiplicacion de matrices.
    Multiply<<<blocks, threads >>>(d_matrixA, d_matrixB, d_matrixC, N);
    AddMatrix<<<blocks, threads>>>(d_matrixC, N, d_result);

    //Copia el resultado de la suma de los elementos de la matriz en la memoria
	cudaMemcpy(h_result, d_result, sizeDouble, cudaMemcpyDeviceToHost);

    //Imprime resultado estimado
    printf("result calc: %.15le\n", N * N * N * a * a);
    
    //Imprime resultado real
    printf("result: %.15le\n", h_result);

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