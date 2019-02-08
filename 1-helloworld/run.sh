#!/bin/sh

case "$1" in

"serial-c")  
    echo "C serial program"
    gcc -o helloworld-c.o helloworld.c
    ./helloworld-c.o
    ;;
"serial-fortran")  
    echo "FORTRAN serial program"
    gfortran -o helloworld-f90.o helloworld.f90
    ./helloworld-f90.o
    ;;
"mpi-c")  
    echo "MPI C program $2 threads"
    mpicc -o helloworld-mpi-c.o helloworld-mpi.c
    mpiexec -np $2 ./helloworld-mpi-c.o
    ;;
"mpi-fortran") 
    echo "MPI FORTRAN program $2 threads"
    mpif90 -o helloworld-mpi-f90.o helloworld-mpi.f90
    mpiexec -np $2 ./helloworld-mpi-f90.o
   ;;
"cuda-c") 
    echo "CUDA C program"
    nvcc -o helloworld-cuda.o helloworld.c
    ./helloworld-cuda.o
   ;;
*) 
    echo "Commands:"
    echo "serial-c : C serial program"
    echo "serial-fortran : FORTRAN serial program"
    echo "mpi-c #threads : MPI C program"
    echo "mpi-fortran #threads: MPI FORTRAN program"
    echo "cuda-c : CUDA C program"
   ;;
esac