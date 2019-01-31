! File: matmult-mpi.f90
!
! Purpose: 
! 
! Input:
! 
! Output:
! 
! Compile: mpif90 -o matmult-mpi-f90.o matmult-mpi.f90
! 
! Run: mpiexec -np 10 ./matmult-mpi-f90.o
! 
! Algorithm:
! 
! Note:

program matmult
    implicit none
    include 'mpif.h'

    INTERFACE 
        subroutine PrintMatrix(array, size)
            REAL, INTENT(IN) :: array(:)
            INTEGER, INTENT(IN) :: size
        end subroutine PrintMatrix

        subroutine FillMatrix(array, size)
            REAL, INTENT(OUT) :: array(:)
            INTEGER, INTENT(IN) :: size
        end subroutine FillMatrix

        subroutine Multiply(MatrixA, MatrixB, MatrixC, sizeX, sizeY)
            REAL, INTENT(IN) :: MatrixA(:)
            REAL, INTENT(IN) :: MatrixB(:)
            REAL, INTENT(OUT) :: MatrixC(:)
            INTEGER, INTENT(IN) :: sizeX
            INTEGER, INTENT(IN) :: sizeY
        end subroutine Multiply

    END INTERFACE

    INTEGER, PARAMETER :: size = 3
    REAL, DIMENSION(size * size) :: matrixA
    REAL, DIMENSION(size * size) :: matrixB
    REAL, DIMENSION(size * size) :: matrixC
    INTEGER :: rank 
    INTEGER :: process
    INTEGER :: processSize
    INTEGER :: ierror
    INTEGER :: status(MPI_STATUS_SIZE)
    INTEGER :: waste
    INTEGER :: n
    INTEGER :: i

    call MPI_INIT(ierror)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, process, ierror)
    call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierror)

    waste = mod(size, process - 1)
    n = size / (process - 1)

    if (rank == 0) then

        call FillMatrix(matrixA, size)
        call FillMatrix(matrixB, size)

    end if

    call MPI_BCAST(matrixA, size * size, MPI_REAL, 0, MPI_COMM_WORLD, ierror)
    call MPI_BCAST(matrixB, size * size, MPI_REAL, 0, MPI_COMM_WORLD, ierror)
    write (*, *) process, rank, n

    if (rank == 0) then
        
        processSize = n

        do i = 1, process - 1, 1

            if (i == process - 1) then
                processSize = waste + n
            end if

            call MPI_RECV(matrixC + ((i - 1) * size * n), processSize * size, MPI_REAL, i, 0, MPI_COMM_WORLD, ierror)

        end do
        call PrintMatrix(matrixA, size)
        call PrintMatrix(matrixB, size)
        call PrintMatrix(matrixC, size)

    else 

        if (rank == process - 1) then
            processSize = waste + n;
        else
            processSize = n;
        end if

        call Multiply(matrixA, matrixB, matrixC, processSize, size)
        call MPI_SEND(matrixC, processSize * size, MPI_REAL, 0, 0, MPI_COMM_WORLD)

    end if

    call MPI_FINALIZE(ierror)

end program matmult

subroutine PrintMatrix(array, size)
    IMPLICIT NONE
    REAL, INTENT(IN) :: array(:)
    INTEGER, INTENT(IN) :: size
    INTEGER :: i, j

    do i = 0, size - 1, 1
        do j = 0, size - 1, 1
            write (*,*) i, j, array(i * size + j) 
        end do
    end do

end subroutine PrintMatrix

subroutine FillMatrix(array, size)
    IMPLICIT NONE
    REAL, INTENT(OUT) :: array(:)
    INTEGER, INTENT(IN) :: size
    INTEGER :: i, j

    do i = 0, size - 1, 1
        do j = 0, size - 1, 1
            array(i * size + j)  = rand(i)
        end do
    end do

end subroutine FillMatrix

subroutine Multiply(MatrixA, MatrixB, MatrixC, sizeX, sizeY)
    IMPLICIT NONE
    REAL, INTENT(IN) :: MatrixA(:)
    REAL, INTENT(IN) :: MatrixB(:)
    REAL, INTENT(OUT) :: MatrixC(:)
    INTEGER, INTENT(IN) :: sizeX
    INTEGER, INTENT(IN) :: sizeY
    INTEGER :: i, j, k
    REAL :: result

    do i = 0, sizeX - 1, 1
        do j = 0, sizeY - 1, 1

            result = 0.0
            do k = 0, sizeY - 1, 1
                result = result + matrixA(i * sizeY + k) * matrixB(k * sizeY + j)
            end do
            matrixC(i * sizeY + j) = result
        
        end do
    end do

end subroutine Multiply