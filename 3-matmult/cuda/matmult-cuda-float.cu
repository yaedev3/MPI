/* File: matmult-cuda-float.cu
 *
 * Purpose: 
 * 
 * Input:
 * 
 * Output:
 * 
 * Compile: nvcc -o matmult-cuda-float.o matmult-cuda-float.cu
 * 
 * Run: ./matmult-cuda-float.o
 * 
 * Algorithm:
 * 
 * Note:
 * 
 * */

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void VecAdd(float* A, float* B, float* C, int N)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x; //indice del vector 
	int ix; //ix indica el renglon 
    int iy; //iy toma valores solo entre 0 a N-1
	float result; //Acumula la suma del renglon por la columna 
	int k; // Iterador 
	
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

// Host code
int main()
{
    //Variables 
	int N;              // Tamaño de la matriz cuadrada.
    int i;              // Indice del renglon.
    int j;              // Indice de la columna.
	size_t size;        // Tamaño total en memoria.
	float* h_A;         // Matriz A en el equipo.
    float* h_B;         // Matriz B en el equipo.
    float* h_C;         // Matriz C (resultado) en el equipo.
    float* d_A;         // Matriz A en la memoria de la GPU.
    float* d_B;         // Matriz B en la memoria de la GPU.
    float* d_C;         // Matriz C (resultado) en la memoria de la GPU.
    int Tam;            // Numero de datos que se manejan
    int NumHilos;       // Hilos por bloque 
    int numBlock;       // Numero de bloques necesario para procesar los datos 

    //Asignacion de variables
    N = 5;
    size = N * sizeof(float) * N;

    //En la memoria del equipo 
	h_A = (float*)malloc(size);
	h_B = (float*)malloc(size);
	h_C = (float*)malloc(size);
	
	//En la memoria de la GPU
	cudaMalloc(&d_A, size);
	cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

	//
    Tam = N * N;
    NumHilos = 1024;
	numBlock = Tam / NumHilos; 

	if(Tam % NumHilos > 0) //Si sobran datos, aumenta los bloques en 1
	    numBlock++;
	
	// LLena los arreglos A y B
	for(i = 0;i < N;i++) //Renglon 
		for(j = 0;j < N;j++) // Columna 
		{
			h_A[i + i * j] = rand()%(i + 1);
			h_B[i + i * j] = rand()%(i + 1);
			//h_A[j + i * N] = j + i * N + 1;
			//h_B[j + i * N] = j + i * N + 1;
		}

	//Copia los arreglos de memoria del CPU a memoria de la GPU 
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	// Invoke kernel
	VecAdd<<<numBlock, NumHilos >>>(d_A, d_B, d_C, N);

    
    //Copea el resultado de la multiplicacion de memoria de la GPU a memoria de la CPU
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	
	//Imprime la matriz A
	printf("Matriz A\n");
	for(i = 0;i < N;i++)
	{
		for(j = 0;j < N;j++)
			printf("%.2f ", h_A[j + i * N]);
		printf("\n");
	}
	
    //Imprime la matriz B
	printf("Matriz B\n");
	for(i = 0;i < N;i++)
	{
		for(j = 0;j < N;j++)
			printf("%.2f ", h_B[j + i * N]);
		printf("\n");
	}
	
    //Imprime la matriz C
	printf("Matriz C\n");
	for(i = 0;i < N;i++)
	{
		for(j = 0;j < N;j++)
			printf("%.2f ", h_C[j + i * N]);
		printf("\n");
	}

	//Libera la memoria utilizada.
	// Free device memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	// Free host memory
	free(h_A);
	free(h_B);
	free(h_C);
}