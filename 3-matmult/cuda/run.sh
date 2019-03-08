#!/bin/bash

if [ ! -d out ]; then
    echo "Creating out directory"
    mkdir out
fi

case "$1" in
    
    "-f")
        nvcc -o out/matmult-cuda-float.o matmult-cuda-float.cu
        time ./out/matmult-cuda-float.o
    ;;
    "-d")
        nvcc -o out/matmult-cuda-double.o matmult-cuda-double.cu
        time ./out/matmult-cuda-double.o
    ;;
    *)
        echo "Commands"
        echo "-f for float CUDA program"
        echo "-d for double CUDA program"
    ;;
esac