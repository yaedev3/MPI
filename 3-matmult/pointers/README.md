# Programa para multiplicar dos matrices cuadradas

## Tabla de contenidos

- [Programa para multiplicar dos matrices cuadradas](#programa-para-multiplicar-dos-matrices-cuadradas)
  - [Tabla de contenidos](#tabla-de-contenidos)
  - [Descripción](#descripci%C3%B3n)
  - [Compilar un programa en especifico](#compilar-un-programa-en-especifico)
  - [Ejecución de un programa en especifico](#ejecuci%C3%B3n-de-un-programa-en-especifico)

## Descripción

Esta version utiliza memoria dinamica usando apuntadores en FORTRAN y C, por lo que puede ser usada para las pruebas debido a que los tamaños de las matrices son variables y se pueden leer de un archivo.

## Compilar un programa en especifico

- Versión serial en c: `gcc -o matmult-c.o matmult.c -lm`
- Versión serial en FORTRAN compilador de intel: `ifort -o matmult-f90.o matmult.f90`
- Versión serial en FORTRAN compilador libre: `gfortran -o matmult-f90.o matmult.f90`
- Versión paralela MPI c: `mpicc -o matmult-mpi-c.o matmult-mpi.c -lm`
- Versión paralela MPI FORTRAN compilador de intel: `mpiifort -O3 -o matmult-mpi-f90.o matmult-mpi.f90`
- Versión paralela MPI FORTRAN compilador libre: `mpif90 -o matmult-mpi-f90.o matmult-mpi.f90`
- Versión parapela CUDA c: `nvcc -o matmult-cuda.o matmult.cu`

## Ejecución de un programa en especifico

- Versión serial en c: `./matmult-c.o $`
- Versión serial en FORTRAN compilador de intel: `./matmult-f90.o`
- Versión serial en FORTRAN compilador libre: `./matmult-f90.o`
- Versión paralela MPI c: `mpiexec -np # ./matmult-mpi-c.o $`
- Versión paralela MPI FORTRAN compilador de intel: `mpiexec -np # ./matmult-mpi-f90.o`
- Versión paralela MPI FORTRAN compilador libre: `mpiexec -np # ./matmult-mpi-f90.o`
- Versión parapela CUDA c: `./matmult-cuda.o $`

**Notas:** # significa el número de procesadores que se quiere utilizar para ejecutar el programa. $ significa el tamaño de la matriz.