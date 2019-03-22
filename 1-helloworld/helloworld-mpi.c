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