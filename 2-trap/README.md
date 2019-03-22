# Trap

## Tabla de contenidos

  - [Descripción](#descripci%C3%B3n)
  - [Requisitos](#requisitos)
  - [Compilar un programa en especifico](#compilar-un-programa-en-especifico)
  - [Ejecución de un programa en especifico](#ejecuci%C3%B3n-de-un-programa-en-especifico)
  - [Archivo parameters.dat](#archivo-parametersdat)
  - [Formato del archivo parameters.dat](#formato-del-archivo-parametersdat)


## Descripción

Calcula la integral usando el método trapezoidal al terminar imprime la estimación del punto `a` al punto `b` usando `n` número de trapecios.

## Requisitos

* `gcc` compilador de c.
* `gfortran`compilador libre de FORTRAN.
* `ifort` compilador de intel de FORTRAN (opcional).
* `mpich`compilador de MPI para FORTRAN y c.
* `nvcc` compilador de CUDA.

## Compilar un programa en especifico

- Versión serial en c: `gcc -o trap-c.o trap.c`
- Versión serial en FORTRAN compilador de intel: `ifort -o trap-f90.o trap.f90`
- Versión serial en FORTRAN compilador libre: `gfortran -o trap-f90.o trap.f90`
- Versión paralela MPI c: `mpicc -o trap-mpi-c.o trap-mpi.c`
- Versión paralela MPI FORTRAN compilador de intel: `mpiifort -O3 -o trap-mpi-f90.o trap-mpi.f90`
- Versión paralela MPI FORTRAN compilador libre: `mpif90 -o trap-mpi-f90.o trap-mpi.f90`
- Versión parapela CUDA c: `nvcc -o trap-cuda.o trap.cu`

## Ejecución de un programa en especifico

- Versión serial en c: `./trap-c.o`
- Versión serial en FORTRAN compilador de intel: `./trap-f90.o`
- Versión serial en FORTRAN compilador libre: `./trap-f90.o`
- Versión paralela MPI c: `mpiexec -np # ./trap-mpi-c.o`
- Versión paralela MPI FORTRAN compilador de intel: `mpiexec -np # ./trap-mpi-f90.o`
- Versión paralela MPI FORTRAN compilador libre: `mpiexec -np # ./trap-mpi-f90.o`
- Versión parapela CUDA c: `./trap-cuda.o`

**Nota:** # significa el número de procesadores que se quiere utilizar para ejecutar el programa.

## Archivo parameters.dat

El archivo parameters.dat sirve para asignar los puntos `a` y `b` así como el `n` número de trapecios. En caso de no contar con el archivo el programa mostrará un mensaje indicando la ausencia del archivo y usará los siguientes valores para el calculo:

* a = 0.0
* b = 3.0
* n = 1024

## Formato del archivo parameters.dat

```
Punto a (doble precisión)
Punto b (doble precisión)
Número de trapecios (entero)

```

## Ejemplo archivo.parameters.dat
```
0.0
3.0
1024
```