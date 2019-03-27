#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Valor constante para llenar la matriz
static double a = 1.0E-10;

void FillMatrix(double *matrixA, double *matrixB, int N);
void Multiply(double *matrixA, double *matrixB, double *matrixC, int N);
double AddMatrix(double *matrix, int N);

int main(int argc, char *argv[])
{
	int N = 5;			   // Dimension de la matriz
	double matrixA[N * N]; // Primera matriz
	double matrixB[N * N]; // Segunda matriz
	double matrixC[N * N]; // Matriz resultado
	double result;		   // Resultado de la suma de la matriz resultado
	double estimation;	 // Estimacion del calculo
	double error;		   // Error encontrado

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

	return 0;
}

// Llena las dos matrices con el valor constante.
void FillMatrix(
	double *matrixA, // Primera matriz
	double *matrixB, // Segunda matriz
	int N			 // Dimension de la matriz
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
	int N			 // Dimension de la matriz
)
{
	int i;		   // Indice del renglon
	int j;		   // Indice de la columna
	int k;		   // Indice de la multiplicacion
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
	int N			// Dimension de la matriz
)
{
	double result; // Resultado de la suma
	int i;		   // Indice del renglon
	int j;		   // Indice de la columna

	result = 0.0;

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			result += matrix[(i * N) + j];

	return result;
}
