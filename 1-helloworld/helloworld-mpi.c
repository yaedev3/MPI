/* File: helloworld-mpi.c
 *
 * Purpose: 
 * 
 * Input:
 * 
 * Output:
 * 
 * Compile: mpicc -o helloworld-mpi-c.o helloworld-mpi.c
 * 
 * Run: mpiexec -np 10 ./helloworld-mpi-c.o
 * 
 * Algorithm:
 * 
 * Note:
 * 
 * */
#include <stdio.h>

#include <mpi.h>

void main()
{
    int rank, process;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process);

    printf("Hello world! from process %d\n", rank);

    MPI_Finalize();
}