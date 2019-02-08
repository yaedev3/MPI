#!/bin/sh

case "$1" in

"-c")  
    gcc -o interpol_v3-c.o interpol_v3.c
    ./interpol_v3-c.o
    ;;
"-f") 
    ifort -o interpol_v3-f90.o interpol_v3.f90 || gfortran -o interpol_v3-f90.o interpol_v3.f90
    ./interpol_v3-f90.o
    ;;
"-mc")  
    echo "MPI C program $2 threads"
    ;;
"-mf") 
    echo "MPI FORTRAN program $2 threads"
   ;;
"-cu") 
    echo "CUDA C program"
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