#!/bin/sh

if [ ! -d out ]; then
    echo "Creating out directory"
    mkdir out
fi

case "$1" in
    
    "-c")
        gcc -o out/trap1-c.o trap1.c
        ./out/trap1-c.o
    ;;
    "-f")
        ifort -o out/trap1-f90.o trap1.f90 || gfortran -o out/trap1-f90.o trap1.f90
        ./out/trap1-f90.o
    ;;
    "-mc")
        echo "MPI C program $2 threads"
        mpicc -o out/trap1-mpi-c.o trap1-mpi.c
        mpiexec -np $2 ./out/trap1-mpi-c.o
    ;;
    "-mf")
        echo "MPI FORTRAN program $2 threads"
        mpif90 -o out/trap1-mpi-f90.o trap1-mpi.f90
        mpiexec -np $2 ./out/trap1-mpi-f90.o
    ;;
    *)
        echo "Commands:"
        echo "-c for C serial program"
        echo "-f for FORTRAN serial program"
        echo "-mc #threads for C MPI program"
        echo "-mf #threads for FORTRAN MPI program"
    ;;
esac