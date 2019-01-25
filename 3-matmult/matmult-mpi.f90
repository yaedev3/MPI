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

        subroutine Multiply(MatrixA, MatrixB, MatrixC, size)
            REAL, INTENT(IN) :: MatrixA(:)
            REAL, INTENT(IN) :: MatrixB(:)
            REAL, INTENT(OUT) :: MatrixC(:)
            INTEGER, INTENT(IN) :: size
        end subroutine Multiply

    END INTERFACE

    INTEGER :: size = 3
    REAL, DIMENSION(25) :: matrixA
    REAL, DIMENSION(25) :: matrixB
    REAL, DIMENSION(25) :: matrixC
    INTEGER :: rank 
    INTEGER :: proccess
    INTEGER :: ierror
    INTEGER :: status(MPI_STATUS_SIZE)
    INTEGER :: waste
    INTEGER :: n
    INTEGER :: i

    call MPI_INIT(ierror)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, proccess, ierror)
    call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierror)

    waste = mod(size, proccess - 1)
    
    if ((size - waste) == 0) then
        n = 0
    else
        n = (size - waste) / (proccess - 1)
        write (*, *) waste, n
    end if 

    if (rank == 0) then

        call FillMatrix(matrixA, size)
        call FillMatrix(matrixB, size)
        call PrintMatrix(matrixA, size)
        call PrintMatrix(matrixB, size)
        write (*, *) waste, n

    end if

    !call Multiply(MatrixA, MatrixB, MatrixC, size)
    !call PrintMatrix(matrixC, size)
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
        do j = 0, size, 1
            array(i * size + j)  = rand(i)
        end do
    end do

end subroutine FillMatrix

subroutine Multiply(MatrixA, MatrixB, MatrixC, size)
    IMPLICIT NONE
    REAL, INTENT(IN) :: MatrixA(:)
    REAL, INTENT(IN) :: MatrixB(:)
    REAL, INTENT(OUT) :: MatrixC(:)
    INTEGER, INTENT(IN) :: size
    INTEGER :: i, j, k
    REAL :: result

    do i = 0, size - 1, 1
        do j = 0, size - 1, 1
            result = 0
            do k = 0, size - 1, 1
                result = result + matrixA(i * size + k) * matrixB(k * size + j)
            end do
            matrixC(i * size + j) = result
        end do
    end do

end subroutine Multiply