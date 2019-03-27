# Programa para multiplicar dos matrices cuadradas

## Tabla de contenidos

- [Programa para multiplicar dos matrices cuadradas](#programa-para-multiplicar-dos-matrices-cuadradas)
  - [Tabla de contenidos](#tabla-de-contenidos)
  - [Descripción](#descripci%C3%B3n)
  - [Compilar un programa en especifico](#compilar-un-programa-en-especifico)
  - [Ejecución de un programa en especifico](#ejecuci%C3%B3n-de-un-programa-en-especifico)

## Descripción

Esta version utiliza memoria estatica por lo que es mas facil de entender e implementar.

## Compilar un programa en especifico

- Versión serial en c: `gcc -o trap-c.o trap.c -lm`
- Versión serial en FORTRAN compilador de intel: `ifort -o trap-f90.o trap.f90`
- Versión serial en FORTRAN compilador libre: `gfortran -o trap-f90.o trap.f90`
- Versión paralela MPI c: `mpicc -o trap-mpi-c.o trap-mpi.c -lm`
- Versión paralela MPI FORTRAN compilador de intel: `mpiifort -O3 -o trap-mpi-f90.o trap-mpi.f90`
- Versión paralela MPI FORTRAN compilador libre: `mpif90 -o trap-mpi-f90.o trap-mpi.f90`
- Versión parapela CUDA c: `nvcc -o trap-cuda.o trap.cu`

## Ejecución de un programa en especifico

- Versión serial en c: `./matmult-c.o`
- Versión serial en FORTRAN compilador de intel: `./matmult-f90.o`
- Versión serial en FORTRAN compilador libre: `./matmult-f90.o`
- Versión paralela MPI c: `mpiexec -np # ./matmult-mpi-c.o`
- Versión paralela MPI FORTRAN compilador de intel: `mpiexec -np # ./matmult-mpi-f90.o`
- Versión paralela MPI FORTRAN compilador libre: `mpiexec -np # ./matmult-mpi-f90.o`
- Versión parapela CUDA c: `./matmult-cuda.o`

**Notas:** # significa el número de procesadores que se quiere utilizar para ejecutar el programa. $ significa el tamaño de la matriz.