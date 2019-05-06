/* File:     mpi_trap1.c
 * Purpose:  Use MPI to implement a parallel version of the trapezoidal 
 *           rule.  In this version the endpoints of the interval and
 *           the number of trapezoids are hardwired.
 *
 * Input:    None.
 * Output:   Estimate of the integral from a to b of f(x)
 *           using the trapezoidal rule and n trapezoids.
 *
 * Compile:  mpicc -g -Wall -o mpi_trap1 mpi_trap1.c
 * Run:      mpiexec -n <number of processes> ./mpi_trap1
 *
 * Algorithm:
 *    1.  Each process calculates "its" interval of
 *        integration.
 *    2.  Each process estimates the integral of f(x)
 *        over its interval using the trapezoidal rule.
 *    3a. Each process != 0 sends its integral to 0.
 *    3b. Process 0 sums the calculations received from
 *        the individual processes and prints the result.
 *
 * Note:  f(x), a, b, and n are all hardwired.
 *
 * IPP:   Section 3.2.2 (pp. 96 and ff.)
 */
#include <stdio.h>
#include <math.h>
#include <time.h>

/* We'll be using MPI routines, definitions, etc. */
#include <mpi.h>

/* Calculate local integral  */
double Trap(double left_endpt, double right_endpt, int trap_count, double base_len);
void OpenFile(double *a, double *b, long *n);
void SaveFile(struct tm *start, struct tm *end, double error, double elapsed, long N, int rank);

int main(int argc, char *argv[])
{
   int my_rank;         //
   int comm_sz;         //
   long n;              //
   int local_n;         //
   int source;          //
   double a;            //
   double b;            //
   double h;            //
   double local_a;      //
   double local_b;      //
   double local_int;    //
   double total_int;    //
   double estimation;   // Estimacion del calculo
   double error;        // Error encontrado
   double elapsed;      // Tiempo que tomo ejecutarse el programa
   time_t t;            // Variable de tiempo
   struct tm *start;    // Hora de inicio
   struct tm *end;      // Hora de termino
   clock_t start_clock; // Tiempo de inicio
   clock_t stop_clock;  // Tiempo de termino

   /* Let the system do what it needs to start up MPI */
   MPI_Init(&argc, &argv);

   /* Get my process rank */
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   /* Find out how many processes are being used */
   MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

   if (my_rank == 0)
   {
      // Inicia la variable de tiempo
      t = time(NULL);

      // Establece la hora de inicio
      start = localtime(&t);
      start_clock = clock();

      OpenFile(&a, &b, &n);
   }

   MPI_Bcast(&a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   MPI_Bcast(&b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   MPI_Bcast(&n, 1, MPI_LONG, 0, MPI_COMM_WORLD);

   h = (b - a) / n;       /* h is the same for all processes */
   local_n = n / comm_sz; /* So is the number of trapezoids  */

   /* Length of each process' interval of
    * integration = local_n*h.  So my interval
    * starts at: */
   local_a = a + my_rank * local_n * h;
   local_b = local_a + local_n * h;
   local_int = Trap(local_a, local_b, local_n, h);

   /* Add up the integrals calculated by each process */
   MPI_Reduce(&local_int, &total_int, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

   //
   if (my_rank == 0)
   {
      // Estimacion del resultado
      estimation = 9.0;

      // Calcula el porcentaje de error.
      error = fabs(total_int - estimation) / estimation * 100.0;

      // Establece la hora en que termino de calcular
      end = localtime(&t);
      stop_clock = clock();

      // Calcula el tiempo que tomo ejecutar el programa
      elapsed = (double)(stop_clock - start_clock) / CLOCKS_PER_SEC;

      // Guarda el resultado en un archivo
      SaveFile(start, end, error, elapsed, n, comm_sz);
   }

   /* Shut down MPI */
   MPI_Finalize();

   return 0;
} /*  main  */

/*------------------------------------------------------------------
 * Function:     Trap
 * Purpose:      Serial function for estimating a definite integral 
 *               using the trapezoidal rule
 * Input args:   left_endpt
 *               right_endpt
 *               trap_count 
 *               base_len
 * Return val:   Trapezoidal rule estimate of integral from
 *               left_endpt to right_endpt using trap_count
 *               trapezoids
 */
double Trap(double left_endpt, double right_endpt, int trap_count, double base_len)
{
   double estimate;
   long i;

   estimate = (pow(left_endpt, 2) + pow(right_endpt, 2)) / 2.0;

   for (i = 1; i <= trap_count - 1; i++)
      estimate += pow(left_endpt + i * base_len, 2);

   estimate = estimate * base_len;

   return estimate;
}

void OpenFile(double *a, double *b, long *n)
{
   FILE *file;
   char *input_file;

   input_file = "parameters.dat";

   file = fopen(input_file, "r");

   fscanf(file, "%lf", a);
   fscanf(file, "%lf", b);
   fscanf(file, "%ld", n);

   *n = (long)pow((double)*n, 2);

   fclose(file);
}
// Crea un archivo de salida con la hora de inicio, de termino y el tiempo que tomo correr el programa
// Asi como el porcentaje de error
void SaveFile(
    struct tm *start, // Hora de inicio
    struct tm *end,   // Hora de termino
    double error,     // Porcentaje de error
    double elapsed,   // Tiempo que paso
    long N,           // Dimension de la matriz
    int rank)
{
   FILE *file;
   char file_name[64];
   char output[50];

   sprintf(file_name, "mpi-%d-gcc-%ld-%d-%d-%d-%d.txt", rank, N, end->tm_mday, end->tm_hour, end->tm_min, end->tm_sec);

   file = fopen(file_name, "w+");

   strftime(output, sizeof(output), "%c", start);
   fprintf(file, "Hora de inicio\n%s\n", output);

   strftime(output, sizeof(output), "%c", end);
   fprintf(file, "Hora de termino\n%s\n", output);

   fprintf(file, "Tiempo de ejecucion\n%.15lf\n", elapsed);

   fprintf(file, "Error\n%.15le\n", error);

   fclose(file);
}