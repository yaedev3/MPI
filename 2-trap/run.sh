#!/bin/bash

if [ ! -d out ]; then
    echo "Creating out directory"
    mkdir out
fi

case "$1" in
    
    "-c")
        icc -O3 -o out/trap-icc.o trap-icc.c -lm || gcc -o out/trap-c.o trap.c -lm
        ./out/trap-c.o
    ;;
    "-f")
        ifort -O3 -o out/trap-f90.o trap.f90 || gfortran -o out/trap-f90.o trap.f90
        ./out/trap-f90.o
    ;;
    "-mc")
        echo "MPI C program $2 threads"
        mpicc -o out/trap-mpi-c.o trap-mpi.c
        mpiexec -np $2 ./out/trap-mpi-c.o
    ;;
    "-mf")
        echo "MPI FORTRAN program $2 threads"
        mpiifort -O3 -o out/trap-mpi-f90.o trap-mpi.f90 || mpif90 -o out/trap-mpi-f90.o trap-mpi.f90
        mpiexec -np $2 ./out/trap-mpi-f90.o
    ;;
    "-cu")
        nvcc -o out/trap-cuda.o trap.cu
        ./out/trap-cuda.o
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