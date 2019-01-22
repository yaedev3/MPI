/* File: helloworld.c
 *
 * Purpose: 
 * 
 * Input:
 * 
 * Output:
 * 
 * Compile: mpicc -o helloworld-mpi.o helloworld-mpi.c
 * 
 * Run: mpiexec -np 10 ./helloworld-mpi.o
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
    int my_rank, comm_sz;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    printf("Hello world! from process %d\n", my_rank);

    MPI_Finalize();
}