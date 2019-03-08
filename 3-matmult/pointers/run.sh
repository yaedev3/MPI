#!/bin/bash

if [ ! -d out ]; then
    echo "Creating out directory"
    mkdir out
fi

case "$1" in
    
    "-c")
        gcc -o out/matmult-c.o matmult.c
        ./out/matmult-c.o
    ;;
    "-f")
        ifort -o out/matmult-f90.o matmult.f90 || gfortran -o out/matmult-f90.o matmult.f90
        ./out/matmult-f90.o
    ;;
    "-mc")
        echo "MPI C program $2 threads"
        mpicc -o out/matmult-mpi-c.o matmult-mpi.c
        mpiexec -np $2 ./out/matmult-mpi-c.o
    ;;
    "-mf")
        echo "MPI FORTRAN program $2 threads"
        mpif90_intel -o out/matmult-mpi-f90.o matmult-mpi.f90 || mpif90 -o out/matmult-mpi-f90.o matmult-mpi.f90
        mpiexec_intel -np $2 ./out/matmult-mpi-f90.o || mpiexec -np $2 ./out/matmult-mpi-f90.o
    ;;
    *)
        echo "Commands:"
        echo "-c for C serial program"
        echo "-f for FORTRAN serial program"
        echo "-mc #threads for C MPI program"
        echo "-mf #threads for FORTRAN MPI program"
    ;;
esac