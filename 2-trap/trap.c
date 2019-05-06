/* File:    trap.c
 * Purpose: Calculate definite integral using trapezoidal 
 *          rule.
 *
 * Input:   a, b, n
 * Output:  Estimate of integral from a to b of f(x)
 *          using n trapezoids.
 *
 * Compile: gcc -g -Wall -o trap trap.c
 * Usage:   ./trap
 *
 * Note:    The function f(x) is hardwired.
 *
 * IPP:     Section 3.2.1 (pp. 94 and ff.) and 5.2 (p. 216)
 */

#include <stdio.h>
#include <math.h>
#include <time.h>

double Trap(double a, double b, long n, double h);
void OpenFile(double *a, double *b, long *n);
void SaveFile(struct tm *start, struct tm *end, double error, double elapsed, long N);

int main(int argc, char *argv[])
{
    double integral;     /* Store result in integral */
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

    h = (b - a) / n;
    integral = Trap(a, b, n, h);

    // Estimacion del resultado
    estimation = 9.0;

    // Calcula el porcentaje de error.
    error = fabs(integral - estimation) / estimation * 100.0;

    // Establece la hora en que termino de calcular
    end = localtime(&t);
    stop_clock = clock();

    // Calcula el tiempo que tomo ejecutar el programa
    elapsed = (double)(stop_clock - start_clock) / CLOCKS_PER_SEC;

    // Guarda el resultado en un archivo
    SaveFile(start, end, error, elapsed, n);

    return 0;
}

/*------------------------------------------------------------------
 * Function:    Trap
 * Purpose:     Estimate integral from a to b of f using trap rule and
 *              n trapezoids
 * Input args:  a, b, n, h
 * Return val:  Estimate of the integral 
 */
double Trap(double a, double b, long n, double h)
{
    double integral;
    long k;

    integral = (a * a + b * b) / 2.0;

    for (k = 1; k <= n - 1; k++)
        integral += (a + k * h) * (a + k * h);

    integral = integral * h;

    return integral;
}

void OpenFile(double *a, double *b, long *n)
{
    FILE *file;
    char *input_file;

    input_file = "parameters.dat";

    file = fopen(input_file, "r");

    if (file == NULL)
    {
        printf("No se encontro el archivo \"parameters.dat\" se usaran parametros por defecto.\n");
        *a = 0.0;
        *b = 3.0;
        *n = 1024;
    }
    else
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

    sprintf(file_name, "serial-c-%ld-%d-%d-%d-%d.txt", N, end->tm_mday, end->tm_hour, end->tm_min, end->tm_sec);

    file = fopen(file_name, "w+");

    strftime(output, sizeof(output), "%c", start);
    fprintf(file, "Hora de inicio\n%s\n", output);

    strftime(output, sizeof(output), "%c", end);
    fprintf(file, "Hora de termino\n%s\n", output);

    fprintf(file, "Tiempo de ejecucion\n%.15lf\n", elapsed);

    fprintf(file, "Error\n%.15le\n", error);

    fclose(file);
}