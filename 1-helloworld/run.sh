#!/bin/bash

if [ ! -d out ]; then
    echo "Creating out directory"
    mkdir out
fi

case "$1" in
    
    "-c")
        gcc -o out/helloworld-c.o helloworld.c
        ./out/helloworld-c.o
    ;;
    "-f")
        ifort -o out/helloworld-f90.o helloworld.f90 || gfortran -o out/helloworld-f90.o helloworld.f90
        ./out/helloworld-f90.o
    ;;
    "-mc")
        echo "MPI C program $2 threads"
        mpicc -o out/helloworld-mpi-c.o helloworld-mpi.c
        mpiexec -np $2 ./out/helloworld-mpi-c.o
    ;;
    "-mf")
        echo "MPI FORTRAN program $2 threads"
        mpif90_intel -o out/helloworld-mpi-f90.o helloworld-mpi.f90 || mpif90 -o out/helloworld-mpi-f90.o helloworld-mpi.f90
        mpiexec_intel -np $2 ./out/helloworld-mpi-f90.o || mpiexec -np $2 ./out/helloworld-mpi-f90.o
    ;;
    "-cu")
        nvcc -o out/helloworld-cuda.o helloworld.cu
        ./out/helloworld-cuda.o
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