! File: matmult.f90
!
! Purpose: 
! 
! Input:
!  
! Output:
! 
! Compile: gfortran -o matmult-f90.o matmult.f90
! 
! Run: ./matmult-f90.o
! 
! Algorithm:
! 
! Note:

program matmult
    implicit none
    
    INTERFACE 
        subroutine PrintMatrix(matrix, size, name)
            REAL, INTENT(IN) :: matrix(:)
            INTEGER, INTENT(IN) :: size
            CHARACTER(len=*), INTENT(IN) :: name
        end subroutine PrintMatrix

        subroutine FillMatrix(matrix, size)
            REAL, INTENT(OUT) :: matrix(:)
            INTEGER, INTENT(IN) :: size
        end subroutine FillMatrix

        subroutine Multiply(MatrixA, MatrixB, MatrixC, size)
            REAL, INTENT(IN) :: MatrixA(:)
            REAL, INTENT(IN) :: MatrixB(:)
            REAL, INTENT(OUT) :: MatrixC(:)
            INTEGER, INTENT(IN) :: size
        end subroutine Multiply

    END INTERFACE

    INTEGER :: size
    REAL, POINTER, DIMENSION(:) :: matrixA
    REAL, POINTER, DIMENSION(:) :: matrixB
    REAL, POINTER, DIMENSION(:) :: matrixC

    size = 5

    allocate(matrixA(size * size))
    allocate(matrixB(size * size))
    allocate(matrixC(size * size))

    call FillMatrix(matrixA, size)
    call FillMatrix(matrixB, size)

    call Multiply(MatrixA, MatrixB, MatrixC, size)

    call PrintMatrix(matrixA, size, 'Matrix A')
    call PrintMatrix(matrixB, size, 'Matrix B')
    call PrintMatrix(matrixC, size, 'Matrix C (result)')

    deallocate(matrixA)
    deallocate(matrixB)
    deallocate(matrixC)

end program matmult

subroutine PrintMatrix(matrix, size, name)
    IMPLICIT NONE
    REAL, INTENT(IN) :: matrix(:)
    INTEGER, INTENT(IN) :: size
    CHARACTER(len=*), INTENT(IN) :: name
    INTEGER :: i
    INTEGER :: j

    write(*, *) name
    do i = 1, size, 1
        do j = 1, size, 1
            write (*,*) i, j, matrix((i - 1) * size + j) 
        end do
    end do

end subroutine PrintMatrix

subroutine FillMatrix(matrix, size)
    IMPLICIT NONE
    REAL, INTENT(OUT) :: matrix(:)
    INTEGER, INTENT(IN) :: size
    INTEGER :: i
    INTEGER :: j

    do i = 1, size, 1
        do j = 1, size, 1
            matrix((i - 1) * size + j)  = rand()
        end do
    end do

end subroutine FillMatrix

subroutine Multiply(MatrixA, MatrixB, MatrixC, size)
    IMPLICIT NONE
    REAL, INTENT(IN) :: MatrixA(:)
    REAL, INTENT(IN) :: MatrixB(:)
    REAL, INTENT(OUT) :: MatrixC(:)
    INTEGER, INTENT(IN) :: size
    INTEGER :: i
    INTEGER :: j
    INTEGER :: k
    REAL :: result

    do i = 1, size, 1
        do j = 1, size, 1
            result = 0.0
            do k = 1, size, 1
                result = result + matrixA((i - 1) * size + k) * matrixB((k - 1) * size + j)
            end do
            matrixC((i - 1) * size + j) = result
        end do
    end do

end subroutine Multiply