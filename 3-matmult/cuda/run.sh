#!/bin/sh

if [ ! -d out ]; then
    echo "Creating out directory"
    mkdir out
fi

case "$1" in

"-f")  
    nvcc -o out/matmult-cuda-float.o matmult-cuda-float.cu
    ./out/matmult-cuda-float.o
    ;;
"-d") 
    nvcc -o out/matmult-cuda-double.o matmult-cuda-double.cu
    ./out/matmult-cuda-double.o
   ;;
"-tf")
# test time in cuda float program  
    nvcc -o out/matmult-cuda-float.o matmult-cuda-float.cu
    time ./out/matmult-cuda-float.o
    ;;
"-td") 
# test time in cuda double program
    nvcc -o out/matmult-cuda-double.o matmult-cuda-double.cu
    time ./out/matmult-cuda-double.o
   ;;
*) 
    echo "-f"
    echo "-d"
    echo "-tf"
    echo "-td"
   ;;
esac