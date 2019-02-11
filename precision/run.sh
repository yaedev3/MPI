#!/bin/sh

if [ ! -d out ]; then
    echo "Creating out directory"
    mkdir out
fi

case "$1" in
    
    "-c")
        gcc -o out/precision-c.o precision.c
        ./out/precision-c.o
    ;;
    "-f")
        gfortran -o out/precision-f90.o precision.f90
        ./out/precision-f90.o
    ;;
    "-mc")
        echo "MPI C program $2 threads"
        mpicc -o out/precision-mpi-c.o precision-mpi.c
        mpiexec -np $2 ./out/precision-mpi-c.o
    ;;
    "-mf")
        echo "MPI FORTRAN program $2 threads"
        mpif90 -o out/precision-mpi-f90.o precision-mpi.f90
        mpiexec -np $2 ./out/precision-mpi-f90.o
    ;;
    "-cu")
        nvcc -o out/precision-cuda.o precision.cu
        ./out/precision-cuda.o
    ;;
    *)
        echo "Commands:"
        echo "-c for C serial program"
        echo "-f for FORTRAN serial program"
        echo "-mc #threads for C MPI program"
        echo "-mf #threads for FORTRAN MPI program"
        echo "-cu for C CUDA program"
    ;;
esac