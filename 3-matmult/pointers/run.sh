#!/bin/bash

if [ ! -d out ]; then
    echo "Creating out directory"
    mkdir out
fi

case "$1" in
    
    "-c")
        icc -o out/matmult-c.o matmult.c -lm  || gcc -o out/matmult-c.o matmult.c -lm
        ./out/matmult-c.o
    ;;
    "-f")
        ifort -o out/matmult-f90.o matmult.f90 || gfortran -o out/matmult-f90.o matmult.f90
        ./out/matmult-f90.o
    ;;
    "-mc")
        echo "MPI C program $2 threads"
        mpiicc -o out/matmult-mpi-c.o matmult-mpi.c -lm || mpicc -o out/matmult-mpi-c.o matmult-mpi.c -lm
        mpiexec -np $2 ./out/matmult-mpi-c.o
    ;;
    "-mf")
        echo "MPI FORTRAN program $2 threads"
        mpiifort -O3 -o out/matmult-mpi-f90.o matmult-mpi.f90 || mpif90 -o out/matmult-mpi-f90.o matmult-mpi.f90
        mpiexec -np $2 ./out/matmult-mpi-f90.o
    ;;
    "-cu")
        nvcc -o out/matmult-cuda.o matmult.cu
        ./out/matmult-cuda.o
    ;;
    "-t")
        icc -o out/matmult-c.o matmult.c -lm  || gcc -o out/matmult-c.o matmult.c -lm
        ifort -o out/matmult-f90.o matmult.f90 || gfortran -o out/matmult-f90.o matmult.f90
        mpicc -o out/matmult-mpi-c.o matmult-mpi.c -lm
        mpiifort -O3 -o out/matmult-mpi-f90.o matmult-mpi.f90 || mpif90 -o out/matmult-mpi-f90.o matmult-mpi.f90
        nvcc -o out/matmult-cuda.o matmult.cu
    ;;
    *)
        echo "Commands:"
        echo "-c for C serial program"
        echo "-f for FORTRAN serial program"
        echo "-mc #threads for C MPI program"
        echo "-mf #threads for FORTRAN MPI program"
        echo "-cu for CUDA"
        echo "-t for compile all programs"
    ;;
esac