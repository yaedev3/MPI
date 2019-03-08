#!/bin/bash

if [ ! -d out ]; then
    echo "Creating out directory"
    mkdir out
fi

case "$1" in
    
    "-c")
        gcc -o out/interpol-c.o interpol.c
        ./out/interpol-c.o
    ;;
    "-f")
        ifort -o out/interpol-f90.o interpol.f90 || gfortran -o out/interpol-f90.o interpol.f90
        ./out/interpol-f90.o
    ;;
    "-mc")
        echo "MPI C program $2 threads"
        mpicc -o out/interpol-mpi-c.o interpol-mpi.c
        mpiexec -np $2 ./out/interpol-mpi-c.o
    ;;
    "-mf")
        echo "MPI FORTRAN program $2 threads"
        mpif90_intel -o out/interpol-mpi-f90.o trap-mpi.f90 || mpicc -o out/interpol-mpi-f90.o interpol-mpi.f90
        mpiexec_intel -np $2 ./out/interpol-mpi-f90.o || mpiexec -np $2 ./out/interpol-mpi-f90.o
    ;;
    "-cu")
        echo "CUDA C program"
        nvcc -o out/interpol-cuda.o interpol.cu
        ./out/interpol-cuda.o
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