#!/bin/sh

case "$1" in

"-f")  
    nvcc -o matmult-cuda-float.o matmult-cuda-float.cu
    ./matmult-cuda-float.o
    ;;
"-d") 
    nvcc -o matmult-cuda-double.o matmult-cuda-double.cu
    ./matmult-cuda-double.o
   ;;
"-tf")
# test time in cuda float program  
    nvcc -o matmult-cuda-float.o matmult-cuda-float.cu
    time ./matmult-cuda-float.o
    ;;
"-td") 
# test time in cuda double program
    nvcc -o matmult-cuda-double.o matmult-cuda-double.cu
    time ./matmult-cuda-double.o
   ;;
*) 
    echo "-f"
    echo "-d"
    echo "-tf"
    echo "-td"
   ;;
esac