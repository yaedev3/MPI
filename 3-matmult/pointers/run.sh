#!/bin/sh

case "$1" in

"serial-c")  
    echo "C serial program"
    gcc -o matmult-c.o matmult.c
    ./matmult-c.o
    ;;
"serial-fortran")  
    echo "FORTRAN serial program"
    gfortran -o matmult-f90.o matmult.f90
    ./matmult-f90.o
    ;;
"mpi-c")  
    echo "MPI C program $2 threads"
    mpicc -o matmult-mpi-c.o matmult-mpi.c
    mpiexec -np $2 ./matmult-mpi-c.o
    ;;
"mpi-fortran") 
    echo "MPI FORTRAN program $2 threads"
    mpif90 -o matmult-mpi-f90.o matmult-mpi.f90
    mpiexec -np 10 ./matmult-mpi-f90.o
   ;;
*) 
    echo "Commands:"
    echo "serial-c : C serial program"
    echo "serial-fortran : FORTRAN serial program"
    echo "mpi-c #threads : MPI C program"
    echo "mpi-fortran #threads: MPI FORTRAN program"
    ;;
esac